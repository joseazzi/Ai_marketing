"""Regression tests for chat-agent explanation issues (demo data, 70% margin, profit, +/-30%)."""
import json
import random

from budget_agent import grounding
from budget_agent.agent import SYSTEM_PROMPT
from budget_agent.tools import TOOLS

OPT_TOOL = next(t for t in TOOLS if t["name"] == "optimize_budget")


# 1. profit-mode change limit is explained to the agent -----------------------------------
def test_change_limit_rule_in_profit_mode(demo_ws):
    demo_ws.set_gross_margin(0.7)
    try:
        out = demo_ws.optimize_budget(100_000, 13, max_change_pct=0.3, goal="max_profit")
        assert "only caps increases" in out["change_limit_rule"]
        # a cut deeper than 30% is allowed and expected in profit mode
        cur, rec = out["current_split"]["allocation"], out["recommended"]["allocation"]
        assert any(rec[c] < 0.7 * cur[c] - 1 for c in cur)
        assert "change_limit_rule" not in demo_ws.optimize_budget(100_000, 13, max_change_pct=0.3)
        assert "change_limit_rule" not in demo_ws.optimize_budget(100_000, 13, goal="max_profit")
    finally:
        demo_ws.set_gross_margin(None)
    assert "only caps increases" in OPT_TOOL["description"]
    assert "only caps increases" in SYSTEM_PROMPT


# 2. the agent may only offer options the tools support -----------------------------------
def test_prompt_forbids_unsupported_options():
    assert "only offer next actions the available tools can actually perform" in SYSTEM_PROMPT
    assert 'goal="max_sales"' in SYSTEM_PROMPT and "no way to apply a change limit to cuts" in SYSTEM_PROMPT


# 3. one consistent source for explaining moves --------------------------------------------
def test_biggest_moves_lists_every_changed_channel_with_direction(demo_ws):
    demo_ws.set_gross_margin(0.7)
    try:
        for kw in (dict(max_change_pct=0.3, goal="max_profit"), dict(max_change_pct=0.3), {}):
            out = demo_ws.optimize_budget(100_000, 13, **kw)
            cur, rec = out["current_split"]["allocation"], out["recommended"]["allocation"]
            moves = {m["channel"]: m for m in out["biggest_moves"]}
            for c in cur:
                d = rec[c] - cur[c]
                if abs(d) > 0.05 * cur[c] + 1:
                    assert c in moves, (kw, c)
                    assert moves[c]["direction"] == ("increase" if d > 0 else "decrease")
                    assert moves[c]["current_allocation"] == round(cur[c], 2)
                    assert moves[c]["recommended_allocation"] == round(rec[c], 2)
                    assert moves[c]["current_marginal_roi"] == round(out["current_split"]["marginal_roi"][c], 3)
            assert "marginal_roi_basis" in out
    finally:
        demo_ws.set_gross_margin(None)
    note = demo_ws.get_channel_performance()["note"]
    assert "RECENT HISTORICAL" in note and "optimize_budget" in note
    assert "use only optimize_budget fields" in SYSTEM_PROMPT


# 4. French number formats -----------------------------------------------------------------
TOOL = [{"sales": 84186.52, "mroi": 0.462, "k": 15812.0, "chg": -4635.52, "share": 0.158, "n": 1500.0}]


def test_grounding_french_formats():
    for text in ["84 186,52 $", "84 186,52 $", "84 186,52 $", "0,462", "15,8K", "-4 635,52", "15,8 %"]:
        g = grounding.check(f"Résultat : {text}.", TOOL)
        assert g["passed"], (text, g)
        assert len(g["grounded"]) == 1, (text, g)  # read as ONE number, not "84" and "186,52"
    assert not grounding.check("Résultat : 84 186,99 $.", TOOL)["passed"]
    assert not grounding.check("Résultat : 0,471.", TOOL)["passed"]


def test_grounding_english_formats_and_lists_unchanged():
    assert grounding.check("Sales $84,186.52, ROI 0.462, $15.8K, -$4,635.52, 15.8%, $1,500.", TOOL)["passed"]
    g = grounding.check("Pick option 1, 2 or 3.", TOOL)
    assert g["passed"] and g["numbers_checked"] == 0  # list items are not glued into 1.2 or 123
    assert grounding.check("$1,500", TOOL)["grounded"] == ["$1,500"]
    assert not grounding.check("$1,501", [{"x": 1.501}])["passed"]  # "$1,501" is 1501, never 1.501


def test_grounding_random_invented_figures_still_rejected(demo_ws):
    outputs = json.loads(json.dumps([demo_ws.get_channel_performance(), demo_ws.optimize_budget(100_000, 13),
                                     demo_ws.check_data()], default=str))
    rng = random.Random(3)

    def fr(x, d):
        return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")

    en = [f"${rng.uniform(100, 250_000):,.0f}" for _ in range(200)]
    fr_ = [fr(rng.uniform(100, 250_000), 0) + " $" for _ in range(200)]
    for fakes in (en, fr_):
        rate = sum(grounding.check(f"Value {f}.", outputs)["passed"] for f in fakes) / len(fakes)
        assert rate < 0.10, rate
