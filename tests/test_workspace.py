from budget_agent.autopilot import run_goal


def test_autopilot_and_learning_loop(demo_ws):
    out = run_goal(demo_ws, 100_000, 13, log=lambda *_: None)
    assert "Recommendation" in out["summary"]
    before = demo_ws.model.trained_weeks
    res = demo_ws.ingest_new_data(demo_weeks=4)
    assert res["weeks_added"] == 4
    assert demo_ws.model.trained_weeks == before + 4
    assert res["refit"]["mode"].startswith("updated")
    assert "recommendation_shift" in res
    assert len(demo_ws.memory["recommendations"]) >= 2


def test_alerts_logged(demo_ws):
    res = demo_ws.check_saturation_alerts(notify=True)
    if res["alerts"]:
        assert "log" in res["delivered_via"]
        assert demo_ws.alerts_path.exists()


def test_profit_mode(demo_ws):
    assert "error" in demo_ws.set_gross_margin(1.5)
    demo_ws.set_gross_margin(None)
    assert "error" in demo_ws.optimize_budget(100_000, 13, goal="max_profit")
    demo_ws.set_gross_margin(0.5)
    try:
        assert demo_ws.breakeven == 2.0
        sales = demo_ws.optimize_budget(100_000, 13)
        profit = demo_ws.optimize_budget(100_000, 13, goal="max_profit")
        assert profit["recommended"]["total_budget"] <= 100_000 + 1
        assert profit["recommended_profit"] >= sales["recommended_profit"] - 1  # profit goal never earns less profit
        perf = demo_ws.get_channel_performance()
        assert perf["breakeven_sales_per_ad_dollar"] == 2.0
        assert "profit_per_ad_dollar" in perf["channels"][0]
    finally:
        demo_ws.set_gross_margin(None)
