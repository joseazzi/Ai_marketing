import json
import os

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["BUDGET_AGENT_WORKSPACE"] = str(tmp_path_factory.mktemp("api"))
    import importlib

    from budget_agent import server
    server = importlib.reload(server)
    server.ws.mmm_maxiter = 25
    c = TestClient(server.app)
    c.server = server
    return c


def test_flow(client):
    s = client.get("/api/state").json()
    assert s["has_data"] is False
    assert client.post("/api/optimize", json={"total_budget": 1000}).status_code == 400  # no data → clear error
    s = client.post("/api/data/demo", json={"seed": 7}).json()
    assert s["weeks"] == 156 and s["demo_weeks_remaining"] > 0
    assert client.get("/api/data").json()["quality"]["ok_to_model"]
    fit = client.post("/api/model/fit", json={}).json()
    assert fit["r2"] > 0.8
    m = client.get("/api/model").json()
    assert m["performance"]["channels"] and m["curves"] and m["fit"]
    opt = client.post("/api/optimize", json={"total_budget": 100000, "n_weeks": 13, "max_change_pct": 0.3}).json()
    assert abs(opt["recommended"]["total_budget"] - 100000) < 1
    wi = client.post("/api/whatif", json={"changes": {"tv": -0.3}}).json()
    assert wi["comparison"]["scenarios"]
    plan = client.post("/api/plan", json={"scenario": "recommended"}).json()
    assert client.get(f"/api/plans/{plan['csv_name']}").status_code == 200
    assert client.get("/api/plans/..%2F..%2Fetc%2Fpasswd").status_code == 404
    ing = client.post("/api/ingest", json={"demo_weeks": 4}).json()
    assert ing["weeks_added"] == 4
    s = client.post("/api/settings", json={"gross_margin": 0.5}).json()
    assert s["breakeven"] == 2.0
    assert "recommended_profit" in client.post("/api/optimize", json={"total_budget": 100000, "goal": "max_profit"}).json()
    client.post("/api/settings", json={"clear_margin": True})


def test_chat_streams_steps_and_answer(client, monkeypatch):
    from budget_agent.agent import BudgetAgent
    from test_agent import FakeClient
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    client.server._agent = BudgetAgent(client.server.ws, client=FakeClient(client.server.ws))
    r = client.post("/api/chat", json={"message": "I have $100K for Q1"})
    events = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    assert [e["tool"] for e in events if e["type"] == "step"] == ["make_plan", "check_data", "optimize_budget", "generate_budget_plan"]
    assert events[-1]["type"] == "answer" and events[-1]["grounding"]["passed"]
    client.server._agent = None


def test_chat_without_key(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client.server._agent = None
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 400 and "OpenAI" in r.json()["detail"]
