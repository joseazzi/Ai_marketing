import json
from types import SimpleNamespace as NS

from budget_agent import grounding
from budget_agent.agent import BudgetAgent


def _tool_msg(*calls):
    return NS(content="", tool_calls=[
        NS(id=f"c{i}", function=NS(name=n, arguments=json.dumps(a))) for i, (n, a) in enumerate(calls)])


class FakeClient:
    """Scripted stand-in for the OpenAI client: plan -> tools -> answer built from tool results."""

    def __init__(self, ws):
        self.ws = ws
        self.turn = 0
        self.chat = NS(completions=NS(create=self.create))

    def create(self, model, messages, tools):
        self.turn += 1
        if self.turn == 1:
            msg = _tool_msg(("make_plan", {"goal": "split $100K for Q1",
                                           "steps": ["check_data", "optimize_budget", "generate_budget_plan"]}),
                            ("check_data", {}))
        elif self.turn == 2:
            msg = _tool_msg(("optimize_budget", {"total_budget": 100000, "n_weeks": 13}))
        elif self.turn == 3:
            msg = _tool_msg(("generate_budget_plan", {}))
        else:
            opt = json.loads([m for m in messages if m["role"] == "tool"][-2]["content"])
            sales = opt["recommended"]["total_incremental_sales"]
            msg = NS(content=f"Spend the $100K as recommended; expected incremental sales ${sales:,.0f}.",
                     tool_calls=None)
        return NS(choices=[NS(message=msg)])


def test_agent_plans_acts_and_is_grounded(demo_ws):
    agent = BudgetAgent(demo_ws, client=FakeClient(demo_ws))
    r = agent.ask("I have $100K for Q1. How should I spend it?")
    assert r.plan and r.plan[0]["steps"][0] == "check_data"
    tools_used = [s.tool for s in r.steps]
    assert tools_used == ["make_plan", "check_data", "optimize_budget", "generate_budget_plan"]
    plan_step = r.steps[-1].result
    assert plan_step["csv_path"].endswith(".csv")
    assert r.grounding["passed"], r.grounding


def test_grounding_flags_invented_numbers():
    res = [{"uplift": 12345.6, "roi": 2.31}]
    g = grounding.check("Uplift of $12.3K at ROI 2.31, plus a made-up 47% gain.", res)
    assert "47%" in g["ungrounded"]
    assert "$12.3K" in g["grounded"] and "2.31" in g["grounded"]


def test_strip_meta_line():
    from budget_agent.agent import _strip_meta
    assert _strip_meta("Rewritten using only numbers from tool outputs.\nSummary") == "Summary"
    assert _strip_meta("Recommended split: ...") == "Recommended split: ..."
