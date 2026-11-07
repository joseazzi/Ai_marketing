"""Regression tests for issues found in code review."""
import json
import random
import threading
from types import SimpleNamespace as NS

import pytest

from budget_agent import grounding
from budget_agent.agent import BudgetAgent
from budget_agent.tools import dispatch
from budget_agent.workspace import Workspace


# 1. grounding must not accept invented figures ------------------------------------------
def test_grounding_rejects_pair_sums_and_most_invented_amounts(demo_ws):
    res = [{"a": 1000.0, "b": 500.0}]
    assert grounding.check("Total $1,500.", res)["ungrounded"] == ["$1,500"]  # no more pair sums
    assert grounding.check("Spend $1,000, i.e. $76.92 per week.", [{"x": 1000.0, "n_weeks": 13}])["passed"]
    assert grounding.check("Saturation is 73% over 13 weeks in Q1 2026.", [{"s": 0.731}])["passed"]
    assert not grounding.check("Plan for 450 customers.", [{"s": 0.731}])["passed"]  # small plain ints are checked

    outputs = json.loads(json.dumps([demo_ws.get_channel_performance(), demo_ws.optimize_budget(100_000, 13),
                                     demo_ws.check_data()], default=str))
    rng = random.Random(0)
    fakes = [f"${rng.uniform(100, 250_000):,.0f}" for _ in range(300)]
    accepted = sum(grounding.check(f"Value {f}.", outputs)["passed"] for f in fakes)
    assert accepted / len(fakes) < 0.10


# 3. the chat must not hold the workspace lock while the LLM thinks -----------------------
def test_workspace_lock_is_free_during_llm_call(demo_ws):
    lock = threading.RLock()
    seen = []

    class Client:
        def __init__(self):
            self.chat = NS(completions=NS(create=self.create))
            self.n = 0

        def create(self, model, messages, tools):
            free = [False]
            t = threading.Thread(target=lambda: free.__setitem__(0, lock.acquire(blocking=False)))
            t.start(); t.join()
            seen.append(free[0])
            if free[0]:
                threading.Thread(target=lock.release).start()  # RLock: must release from the acquiring thread
            self.n += 1
            if self.n == 1:
                return NS(choices=[NS(message=NS(content="", tool_calls=[
                    NS(id="c1", function=NS(name="check_data", arguments="{}"))]))])
            return NS(choices=[NS(message=NS(content="Done.", tool_calls=None))])

    BudgetAgent(demo_ws, client=Client(), tool_lock=lock).ask("check")
    assert seen and all(seen)


# 4. gross margin survives a restart -----------------------------------------------------
def test_gross_margin_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("GROSS_MARGIN", "0.4")
    ws = Workspace(tmp_path)
    assert ws.gross_margin == 0.4                   # from .env
    ws.set_gross_margin(0.55)
    assert Workspace(tmp_path).gross_margin == 0.55  # saved value wins over .env
    ws.set_gross_margin(None)
    assert Workspace(tmp_path).gross_margin is None  # clearing it also persists


# 5. ingest never stops half-way ---------------------------------------------------------
def test_ingest_survives_infeasible_last_goal(tmp_path):
    ws = Workspace(tmp_path, mmm_maxiter=15)
    ws.init_demo()
    ws.fit_model()
    ws.optimize_budget(100_000, 13)
    ws.last_goal["min_spend"] = {"tv": 90_000, "radio": 90_000}  # becomes impossible
    weeks = len(ws.df)
    out = ws.ingest_new_data(demo_weeks=2)
    assert out["weeks_added"] == 2 and len(ws.df) == weeks + 2
    assert ws.model.trained_weeks == weeks + 2
    assert "reoptimize_error" in out and "delivered_via" in out


# 6. the LLM cannot read arbitrary files -------------------------------------------------
def test_ingest_csv_path_is_restricted(demo_ws):
    out = dispatch(demo_ws, "ingest_new_data", {"csv_path": "/etc/passwd"})
    assert "error" in out and "csv_path" in out["error"]


# 7. agent history stays bounded ---------------------------------------------------------
def test_agent_history_is_bounded(demo_ws):
    class Client:
        def __init__(self):
            self.chat = NS(completions=NS(create=self.create))
            self.n = 0

        def create(self, model, messages, tools):
            self.n += 1
            if self.n % 2:
                return NS(choices=[NS(message=NS(content="", tool_calls=[
                    NS(id=f"c{self.n}", function=NS(name="get_channel_performance", arguments="{}"))]))])
            return NS(choices=[NS(message=NS(content="ok", tool_calls=None))])

    agent = BudgetAgent(demo_ws, client=Client(), max_turns_kept=3, old_tool_chars=500)
    for i in range(10):
        agent.ask(f"question {i}")
    msgs = agent.messages
    assert sum(m["role"] == "user" for m in msgs) == 3
    assert sum(m["role"] == "system" and m["content"].startswith("[workspace state]") for m in msgs) == 1
    assert all(len(m["content"]) < 600 for m in msgs[:-3] if m["role"] == "tool")


# 8. UI-facing semantics -----------------------------------------------------------------
def test_profit_mode_can_cut_below_max_change(demo_ws):
    demo_ws.set_gross_margin(0.5)
    try:
        out = demo_ws.optimize_budget(100_000, 13, max_change_pct=0.3, goal="max_profit")
        cur, rec = out["current_split"]["allocation"], out["recommended"]["allocation"]
        cut = [c for c in cur if rec[c] < 0.7 * cur[c] - 1]
        assert cut, "an unprofitable channel should be allowed to drop more than 30%"
        assert all(rec[c] <= 1.3 * cur[c] + 1 for c in cur)  # increases are still capped
        statuses = {c["status"] for c in demo_ws.get_channel_performance()["channels"]}
        assert "below break-even" in statuses
    finally:
        demo_ws.set_gross_margin(None)


@pytest.fixture(autouse=True)
def _no_alert_network(monkeypatch):
    for k in ("SLACK_WEBHOOK_URL", "ALERT_WEBHOOK_URL", "SMTP_HOST"):
        monkeypatch.delenv(k, raising=False)


def test_fallback_keeps_original_goal(tmp_path):
    ws = Workspace(tmp_path, mmm_maxiter=15)
    ws.init_demo()
    ws.fit_model()
    ws.set_gross_margin(0.5)
    ws.optimize_budget(100_000, 13, goal="max_profit")
    ws.gross_margin = None  # profit goal becomes impossible for this ingest
    out = ws.ingest_new_data(demo_weeks=2)
    assert "goal_fallback" in out and "new_recommendation" in out
    assert ws.last_goal["goal"] == "max_profit"
    assert Workspace(tmp_path).last_goal["goal"] == "max_profit"  # also persisted


def test_reset_waits_for_running_answer(demo_ws):
    import time
    started, release = threading.Event(), threading.Event()

    class Slow:
        def __init__(self):
            self.chat = NS(completions=NS(create=self.create))

        def create(self, model, messages, tools):
            started.set(); release.wait(5)
            return NS(choices=[NS(message=NS(content="answer", tool_calls=None))])

    agent = BudgetAgent(demo_ws, client=Slow())
    t = threading.Thread(target=agent.ask, args=("q",)); t.start()
    started.wait(5)
    r = threading.Thread(target=agent.reset); r.start()
    time.sleep(0.2)
    assert r.is_alive()  # reset is waiting, not editing the history mid-answer
    release.set(); t.join(5); r.join(5)
    assert agent.messages == agent.messages[:1]
