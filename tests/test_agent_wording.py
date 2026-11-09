"""Regression tests for chat-agent wording issues (demo data, 70% gross margin), using a fake OpenAI client."""
import json
from types import SimpleNamespace as NS

from budget_agent.agent import SYSTEM_PROMPT, BudgetAgent
from budget_agent.tools import TOOLS


class ScriptedClient:
    """Fake OpenAI client: calls the given tools, then answers with ``answer(tool_results)``."""

    def __init__(self, calls, answer):
        self.calls, self.answer, self.n, self.seen = calls, answer, 0, []
        self.chat = NS(completions=NS(create=self.create))

    def create(self, model, messages, tools):
        self.seen.append(messages)
        if self.n < len(self.calls):
            name, args = self.calls[self.n]
            self.n += 1
            return NS(choices=[NS(message=NS(content="", tool_calls=[
                NS(id=f"c{self.n}", function=NS(name=name, arguments=json.dumps(args)))]))])
        results = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        return NS(choices=[NS(message=NS(content=self.answer(results), tool_calls=None))])


# 1. loss warning ---------------------------------------------------------------------
def test_profit_note_when_sales_plan_spends_past_break_even(demo_ws):
    demo_ws.set_gross_margin(0.7)
    try:
        out = demo_ws.optimize_budget(100_000, 13)
        note = out["profit_note"]
        assert "below break-even $1.43" in note and "lose money" in note and "goal='max_profit' would give" in note
        assert "profit_note" not in demo_ws.optimize_budget(100_000, 13, goal="max_profit")

        # the agent can quote the note and pass the number check
        def answer(results):
            ret = note.split("next $1 returns ")[1].split(" in sales")[0]  # e.g. "$1.24" or "between $a and $b"
            return (f"Recommended split below. Note: at this spend the next $1 returns {ret} in sales, below the "
                    "$1.43 break-even at a 70% margin, so the last dollars lose money; a profit plan would "
                    + note.split("would give ")[1])
        client = ScriptedClient([("optimize_budget", {"total_budget": 100000, "n_weeks": 13})], answer)
        r = BudgetAgent(demo_ws, client=client).ask("I have $100K for Q1. How should I spend it?")
        assert r.grounding["passed"], r.grounding
    finally:
        demo_ws.set_gross_margin(None)
    assert "profit_note" not in demo_ws.optimize_budget(100_000, 13)  # no margin -> no note
    assert "PROFIT NOTE: if optimize_budget returns profit_note, always state it" in SYSTEM_PROMPT


# 2 + 3. prompt rules reach the model ---------------------------------------------------
def test_prompt_rules_sent_to_model(demo_ws):
    client = ScriptedClient([], lambda r: "ok")
    BudgetAgent(demo_ws, client=client).ask("hi")
    system = client.seen[0][0]["content"]
    # 2. only tool-backed actions, no scheduling/handoff/deployment
    assert "must map to one tool call" in system
    assert "Never offer or promise scheduling, handoff, deployment" in system
    # 3. the plan comes from the response curves, not from alerts or historical figures
    assert "maximises the model's fitted response curves" in system
    assert "never feed the optimiser" in system
    assert 'Never say alerts or historical channel figures were "used in"' in system
    # 4. reliability answer checklist
    for part in ("data period", "holdout R2 and MAPE", "not measured by experiments", "is_simulated_demo"):
        assert part in system


# 4. "based on what?" has the facts to quote ---------------------------------------------
def test_reliability_facts_are_available_and_quotable(demo_ws):
    q = demo_ws.check_data()
    assert q["is_simulated_demo"] is True and q["data_source"].startswith("simulated")
    fit = demo_ws.fit_model()
    h = fit["holdout_accuracy"]
    assert h["test_weeks"] == 13 and 0 < h["holdout_r2"] <= 1 and 0 < h["holdout_mape"] < 1
    assert fit["data_period"]["weeks"] == len(demo_ws.df)
    assert "not measured by experiments" in fit["how_effects_are_estimated"]
    assert demo_ws.fit_model()["holdout_accuracy"] == h  # cached while the data is unchanged

    def answer(results):
        f = next(r for r in results if "holdout_accuracy" in r)
        hh = f["holdout_accuracy"]
        return (f"It is based on {f['data_period']['weeks']} weeks of simulated demo data "
                f"({f['data_period']['start']} to {f['data_period']['end']}). In-sample R² is {f['r2']:.2f}; on the "
                f"last 13 weeks, hidden from training, R² is {hh['holdout_r2']:.2f} with {hh['holdout_mape'] * 100:.1f}% "
                f"error. Channel effects are estimated from past variation, not measured.")
    client = ScriptedClient([("fit_model", {})], answer)
    r = BudgetAgent(demo_ws, client=client).ask("What is this recommendation based on?")
    assert r.grounding["passed"], (r.answer, r.grounding)
    assert "holdout_accuracy" in next(t for t in TOOLS if t["name"] == "fit_model")["description"]
