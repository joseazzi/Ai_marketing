import pytest

from budget_agent.optimizer import current_allocation, evaluate_plan, optimize, what_if


def test_optimizer_beats_current_and_respects_budget(demo_ws):
    m = demo_ws.model
    best = optimize(m, 100_000, 13)
    cur = evaluate_plan(m, current_allocation(m, 100_000, df=demo_ws.df), 13)
    assert abs(best.total_budget - 100_000) < 1
    assert best.total_sales >= cur.total_sales - 1e-6


def test_optimizer_constraints(demo_ws):
    m = demo_ws.model
    p = optimize(m, 100_000, 13, min_spend={"tv": 40_000}, max_spend={"instagram": 5_000})
    assert p.allocation["tv"] >= 40_000 - 1
    assert p.allocation["instagram"] <= 5_000 + 1
    with pytest.raises(ValueError):
        optimize(m, 10_000, 13, min_spend={"tv": 20_000})


def test_what_if_cut(demo_ws):
    m = demo_ws.model
    base = current_allocation(m, 100_000, df=demo_ws.df)
    p = what_if(m, base, 13, {"tv": -0.3})
    assert abs(p.allocation["tv"] - 0.7 * base["tv"]) < 1e-6
    r = what_if(m, base, 13, {"tv": -0.3}, reallocate=True)
    assert abs(r.total_budget - 100_000) < 1
