"""Regression tests: grounding context (user figures, earlier turns, parameters, ratios) and what-if wording.

Uses a scripted fake OpenAI client, as in tests/test_review_fixes.py.
"""
import json
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd
import pytest

from budget_agent import grounding
from budget_agent.agent import SYSTEM_PROMPT, BudgetAgent
from budget_agent.data import load_csv, quality_report
from budget_agent.workspace import Workspace


class Script:
    """Fake OpenAI client. Each turn is (tool calls, answer_fn(tool_results_of_whole_conversation))."""

    def __init__(self, turns):
        self.turns, self.t, self.i = turns, 0, 0
        self.chat = NS(completions=NS(create=self.create))

    def create(self, model, messages, tools):
        calls, answer = self.turns[self.t]
        if self.i < len(calls):
            name, args = calls[self.i]
            self.i += 1
            return NS(choices=[NS(message=NS(content="", tool_calls=[
                NS(id=f"c{self.t}-{self.i}", function=NS(name=name, arguments=json.dumps(args)))]))])
        self.t, self.i = self.t + 1, 0
        return NS(choices=[NS(message=NS(content=answer(), tool_calls=None))])


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    w = Workspace(tmp_path_factory.mktemp("gc"), mmm_maxiter=15)
    w.init_demo()
    w.fit_model()
    return w


# 1. the user's own numbers ----------------------------------------------------------------
def test_user_figures_are_grounded(ws):
    msg = "I have $10,000 for 13 weeks with at least $50,000 on TV."
    answer = ("Your $10,000 over 13 weeks is about $769.23 per week, but the $50,000 TV minimum is larger "
              "than the whole budget, so this can't be planned. Lower the TV minimum or raise the budget.")
    r = BudgetAgent(ws, client=Script([([], lambda: answer)])).ask(msg)
    assert r.grounding["passed"], r.grounding
    assert set(r.grounding["grounded"]) == {"$10,000", "$769.23", "$50,000"}
    assert not grounding.check("Your $12,000 budget", [], user_texts=[msg])["passed"]


# 2. tool results from earlier turns -------------------------------------------------------
def test_earlier_turn_figures_stay_grounded_even_when_truncated(ws):
    ws.set_gross_margin(0.7)
    try:
        holder = {}

        def turn1():
            holder["opt"] = ws.scenarios["recommended"].to_dict()
            return "Here is the profit plan."

        def turn2():
            p = holder["opt"]
            a = p["allocation"]
            return (f"On that plan, TV was ${a['tv']:,.2f} and Google Ads ${a['google_ads']:,.2f}; "
                    f"incremental sales ${p['total_incremental_sales']:,.2f} at ROI {p['roi']:.3f}.")

        agent = BudgetAgent(ws, client=Script([
            ([("optimize_budget", {"total_budget": 100000, "n_weeks": 13, "max_change_pct": 0.3,
                                    "goal": "max_profit"})], turn1),
            ([], turn2), ([], lambda: "ok"), ([], turn2)]), old_tool_chars=80, max_turns_kept=4)
        agent.ask("I have $100K for 13 weeks. Maximize profit, at most 30% change.")
        r = agent.ask("What were the TV and Google Ads amounts on that plan?")
        assert r.grounding["passed"], r.grounding
        agent.ask("thanks")
        assert any("truncated" in m["content"] for m in agent.messages if m["role"] == "tool")
        r = agent.ask("Remind me of TV and Google Ads on that plan?")  # history truncated, still grounded
        assert r.grounding["passed"], r.grounding
        assert len(agent.grounding_turns) <= 4
    finally:
        ws.set_gross_margin(None)


# 3. parameter examples are not figures ----------------------------------------------------
def test_parameter_examples_are_not_checked():
    text = ('Next: run `run_what_if` with changes={"tv": -0.1}, or run_what_if(changes={"radio": 0.25}); '
            'or try `max_change_pct=0.3`.')
    g = grounding.check(text, [{"x": 1.0}])
    assert g["passed"] and g["numbers_checked"] == 0
    assert not grounding.check("TV return drops to 0.37 (from 0.61).", [{"x": 1.0}])["passed"]  # prose still checked


# 4. ratios must be copied, not re-estimated -----------------------------------------------
def test_ratios_use_written_precision_only():
    tool = [{"roi": {"a": 1.505, "b": 1.788, "c": 1.263, "d": 1.139}}]
    assert grounding.check("ROI 1.51, 1.5, 1.79, 1.26, 1.14 and 1.3", tool)["passed"]
    g = grounding.check("ROI 1.53, 1.78, 1.25, 1.15", tool)
    assert g["ungrounded"] == ["1.53", "1.78", "1.25", "1.15"]
    assert "never re-estimate" in SYSTEM_PROMPT


# 5. what-if verdict -----------------------------------------------------------------------
def test_what_if_verdict_and_breakeven(ws):
    ws.set_gross_margin(0.7)
    try:
        ws.optimize_budget(100_000, 13, max_change_pct=0.3, goal="max_profit")
        w = ws.run_what_if({"tv": -0.2}, base="recommended", reallocate=True)
        assert w["verdict"] == "worse" and w["verdict_metric"] == "profit" and w["verdict_change"] < 0
        assert w["channels_below_breakeven"] and all(c["marginal_roi"] < 1 / 0.7 for c in w["channels_below_breakeven"])
        assert "never beat it" in w["base_note"]
        assert "tv" in w["channels_changed"]
        tiny = ws.run_what_if({"tv": -0.0001}, base="recommended")
        assert tiny["verdict"] == "about the same"
    finally:
        ws.set_gross_margin(None)
    s = ws.run_what_if({"tv": -0.3}, base="current split", reallocate=True, total_budget=100_000, n_weeks=13)
    assert s["verdict_metric"] == "sales" and s["verdict"] in ("better", "worse", "about the same")
    assert "channels_below_breakeven" not in s
    assert "start with run_what_if's verdict" in SYSTEM_PROMPT


# 6. profit_note names the losing channels -------------------------------------------------
def test_profit_note_names_channels(ws):
    ws.set_gross_margin(0.7)
    try:
        note = ws.optimize_budget(100_000, 13, max_change_pct=0.3)["profit_note"]
        some_pay = "still pay off" in note
        assert some_pay or note.startswith("On every channel")
        if some_pay:
            assert "tv $" in note and "between" not in note
        note_all = ws.optimize_budget(100_000, 13)["profit_note"]  # no limits: every channel ends at the same return
        assert note_all.startswith("On every channel")
    finally:
        ws.set_gross_margin(None)


# 7. wording rules ---------------------------------------------------------------------------
def test_wording_rules_in_prompt():
    assert 'never call them "historical"' in SYSTEM_PROMPT
    assert "Always write sales and profit amounts with $" in SYSTEM_PROMPT


# 8. moderate correlation is reported --------------------------------------------------------
def test_moderate_and_high_correlation_warnings():
    real = quality_report(load_csv("data/real_apparel_brand_usd.csv"))
    assert any("Moderately correlated" in w and "meta_instagram" in w for w in real["warnings"])
    rng = np.random.default_rng(0)
    base = rng.lognormal(0, 0.5, 120)
    df = pd.DataFrame({"date": pd.date_range("2022-01-03", periods=120, freq="7D"),
                       "sales": rng.normal(100, 5, 120),
                       "spend_a": base, "spend_b": base * 1.01 + rng.normal(0, 0.01, 120),
                       "spend_c": rng.lognormal(0, 0.5, 120)})
    assert any("Highly correlated" in w for w in quality_report(df)["warnings"])
