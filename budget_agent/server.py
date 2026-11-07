"""HTTP API for the React front end (and anything else that wants to drive the agent).

    python -m budget_agent.server            # http://localhost:8000  (serves the built React app too)

All state lives in one Workspace (folder ``BUDGET_AGENT_WORKSPACE``, default ``workspace``).
"""
from __future__ import annotations

import io
import json
import math
import os
import queue
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import data as D
from .env import load_dotenv
from .workspace import Workspace

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "frontend" / "dist"
REAL_DATA = ROOT / "data" / "real_apparel_brand_usd.csv"

app = FastAPI(title="Budget Agent API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])

ws = Workspace(os.getenv("BUDGET_AGENT_WORKSPACE", "workspace"))
lock = threading.RLock()  # the model/workspace is not thread-safe
_agent = None


# --------------------------------------------------------------------------- #
def clean(o):
    """Make numpy/pandas values JSON-safe (NaN/inf -> None)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return o


def records(df: pd.DataFrame) -> list[dict]:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].dt.strftime("%Y-%m-%d")
    return clean(df.to_dict(orient="records"))


def run(fn, *a, **k):
    with lock:
        try:
            out = fn(*a, **k)
        except (RuntimeError, ValueError) as e:
            raise HTTPException(400, str(e))
    if isinstance(out, dict) and "error" in out:
        raise HTTPException(400, out["error"])
    return clean(out)


def agent():
    global _agent
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(400, "No OpenAI key. Add OPENAI_API_KEY to .env or set it in Settings.")
    if _agent is None:
        from .agent import BudgetAgent
        _agent = BudgetAgent(ws, tool_lock=lock)
    return _agent


# --------------------------------------------------------------------------- #
# State & settings
# --------------------------------------------------------------------------- #
@app.get("/api/state")
def state():
    with lock:
        df, m = ws.df, ws.model
        return clean({
            "has_data": df is not None,
            "weeks": len(df) if df is not None else 0,
            "start": str(df[D.DATE_COL].min().date()) if df is not None else None,
            "end": str(df[D.DATE_COL].max().date()) if df is not None else None,
            "channels": D.channels_of(df) if df is not None else [],
            "model": None if m is None else {"trained_weeks": m.trained_weeks, **m.fit_metrics,
                                             "stale": df is not None and m.trained_weeks != len(df)},
            "gross_margin": ws.gross_margin,
            "breakeven": ws.breakeven,
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "openai_model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            "demo_weeks_remaining": ws.demo_weeks_remaining(),
            "scenarios": list(ws.scenarios),
            "last_goal": ws.last_goal,
            "real_data_available": REAL_DATA.exists(),
        })


class Settings(BaseModel):
    openai_api_key: str | None = None
    openai_model: str | None = None
    gross_margin: float | None = None
    clear_margin: bool = False


@app.post("/api/settings")
def settings(s: Settings):
    global _agent
    if s.openai_api_key:
        os.environ["OPENAI_API_KEY"] = s.openai_api_key.strip()
        _agent = None
    if s.openai_model:
        os.environ["OPENAI_MODEL"] = s.openai_model.strip()
        _agent = None
    if s.clear_margin:
        run(ws.set_gross_margin, None)
    elif s.gross_margin is not None:
        run(ws.set_gross_margin, s.gross_margin)
    return state()


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
class Demo(BaseModel):
    seed: int = 7


@app.post("/api/data/demo")
def load_demo(d: Demo):
    run(ws.init_demo, seed=d.seed)
    return state()


@app.post("/api/data/real")
def load_real():
    run(ws.set_data, D.load_csv(REAL_DATA), source="real apparel brand (Conjura)")
    return state()


@app.post("/api/data/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        df = D.load_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {e}")
    run(ws.set_data, df, source=file.filename)
    return state()


@app.get("/api/data")
def data_view():
    with lock:
        if ws.df is None:
            raise HTTPException(400, "No data loaded.")
        q = D.quality_report(ws.df)
        cols = [D.DATE_COL, D.TARGET_COL] + [D.SPEND_PREFIX + c for c in q["channels"]]
        series = ws.df[cols].rename(columns=lambda c: c.replace(D.SPEND_PREFIX, ""))
        return {"quality": clean(q), "series": records(series)}


@app.get("/api/template.csv")
def template():
    buf = io.StringIO()
    D.simulate(weeks=8).data.to_csv(buf, index=False)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=template.csv"})


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Fit(BaseModel):
    recency_halflife_weeks: float | None = None
    from_scratch: bool = True


@app.post("/api/model/fit")
def fit(f: Fit):
    return run(ws.fit_model, recency_halflife_weeks=f.recency_halflife_weeks, from_scratch=f.from_scratch)


@app.get("/api/model")
def model_view():
    with lock:
        if ws.model is None:
            raise HTTPException(400, "Model not trained yet.")
        m, df = ws.model, ws.df
        pred = pd.DataFrame({"date": df[D.DATE_COL], "actual": df[D.TARGET_COL], "predicted": m.predict(df)})
        dec = m.decompose(df)
        curves = m.response_curves(points=40)
        current = {ch: {"weekly_spend": m.spend_stats[ch]["last13_mean"],
                        "weekly_sales": float(m.params[ch].response(m.spend_stats[ch]["last13_mean"]))}
                   for ch in m.channels}
        return {
            "metrics": clean(m.fit_metrics), "trained_weeks": m.trained_weeks,
            "performance": run(ws.get_channel_performance),
            "fit": records(pred), "decomposition": records(dec),
            "curves": records(curves), "current_points": clean(current),
        }


# --------------------------------------------------------------------------- #
# Optimizer & scenarios
# --------------------------------------------------------------------------- #
class Optimize(BaseModel):
    total_budget: float
    n_weeks: int = 13
    min_spend: dict[str, float] | None = None
    max_spend: dict[str, float] | None = None
    max_change_pct: float | None = None
    goal: str = "max_sales"


@app.post("/api/optimize")
def optimize(o: Optimize):
    return run(ws.optimize_budget, o.total_budget, o.n_weeks, o.min_spend or None, o.max_spend or None,
               o.max_change_pct, o.goal)


class WhatIf(BaseModel):
    changes: dict[str, float]
    base: str = "current split"
    reallocate: bool = False
    total_budget: float | None = None
    n_weeks: int | None = None


@app.post("/api/whatif")
def whatif(w: WhatIf):
    out = run(ws.run_what_if, w.changes, w.base, w.reallocate, w.total_budget, w.n_weeks)
    out["comparison"] = run(ws.compare_scenarios)
    return out


@app.get("/api/scenarios")
def scenarios():
    if not ws.scenarios:
        return {"scenarios": []}
    return run(ws.compare_scenarios)


class GenPlan(BaseModel):
    scenario: str = "recommended"


@app.post("/api/plan")
def gen_plan(g: GenPlan):
    out = run(ws.generate_budget_plan, g.scenario)
    out["csv_name"] = Path(out["csv_path"]).name
    out["brief_name"] = Path(out["brief_path"]).name
    out["brief_markdown"] = Path(out["brief_path"]).read_text()
    return out


@app.get("/api/plans")
def list_plans():
    files = sorted((ws.root / "plans").glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"brief_name": p.name, "csv_name": p.with_suffix(".csv").name,
             "title": p.read_text().splitlines()[0].lstrip("# ")} for p in files[:20]]


@app.get("/api/plans/{name}")
def get_plan(name: str):
    p = (ws.root / "plans" / name).resolve()
    if p.parent != (ws.root / "plans").resolve() or not p.exists():
        raise HTTPException(404, "Not found")
    return FileResponse(p, filename=p.name)


# --------------------------------------------------------------------------- #
# Monitoring
# --------------------------------------------------------------------------- #
class Ingest(BaseModel):
    demo_weeks: int = 4


@app.post("/api/ingest")
def ingest(i: Ingest):
    return run(ws.ingest_new_data, demo_weeks=i.demo_weeks)


@app.post("/api/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        rows = D.load_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {e}")
    return run(ws.ingest_new_data, new_rows=rows)


@app.post("/api/alerts")
def alerts():
    return run(ws.check_saturation_alerts, notify=True)


@app.get("/api/history")
def history():
    return run(ws.get_history, limit=50)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
class Evaluate(BaseModel):
    budget: float = 100_000


@app.post("/api/evaluate")
def evaluate(e: Evaluate):
    from .evaluation import business_value, ground_truth_recovery, holdout_accuracy
    with lock:
        if ws.df is None:
            raise HTTPException(400, "No data loaded.")
        acc = holdout_accuracy(ws.df)
        series = records(acc.pop("series"))
        bv = business_value(ws.model, e.budget, 13, df=ws.df) if ws.model else None
    gt = ground_truth_recovery()
    return clean({"accuracy": {**acc, "series": series},
                  "business_value": {k: v for k, v in (bv or {}).items() if k not in ("current", "optimized")} or None,
                  "validation": gt})


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
class Autopilot(BaseModel):
    total_budget: float = 100_000
    n_weeks: int = 13


@app.post("/api/autopilot")
def autopilot(a: Autopilot):
    from .autopilot import run_goal
    logs: list[str] = []
    with lock:
        try:
            res = run_goal(ws, a.total_budget, a.n_weeks, log=logs.append)
        except (RuntimeError, ValueError) as e:
            raise HTTPException(400, str(e))
    return clean({"log": logs, "plan": res["plan"], "summary": res["summary"],
                  "optimize": res["trace"].get("optimize_budget"),
                  "plan_files": res["trace"].get("generate_budget_plan")})


class Chat(BaseModel):
    message: str


@app.post("/api/chat")
def chat(c: Chat):
    """Streams NDJSON events: {"type":"step",...} for each tool call, then {"type":"answer",...}."""
    ag = agent()
    q: queue.Queue = queue.Queue()

    def on_step(s):
        q.put({"type": "step", "tool": s.tool, "args": s.args, "seconds": round(s.seconds, 2),
               "error": s.result.get("error") if isinstance(s.result, dict) else None,
               "result": s.result})

    def work():
        try:
            r = ag.ask(c.message, on_step=on_step)  # holds the workspace lock only while a tool runs
            q.put({"type": "answer", "answer": r.answer, "plan": r.plan[-1] if r.plan else None,
                   "grounding": r.grounding, "revised": r.revised})
        except Exception as e:  # surface OpenAI/config errors to the UI
            q.put({"type": "error", "error": f"{type(e).__name__}: {e}"})
        q.put(None)

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while (ev := q.get()) is not None:
            yield json.dumps(clean(ev), default=str) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.post("/api/chat/reset")
def chat_reset():
    if _agent is not None:
        _agent.reset()
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Front end
# --------------------------------------------------------------------------- #
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        if path.startswith("api/"):
            raise HTTPException(404, "Not found")
        f = (DIST / path).resolve()
        if path and f.is_file() and DIST.resolve() in f.parents:
            return FileResponse(f)
        return FileResponse(DIST / "index.html")


def main():
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    print(f"\n  Budget Agent running at http://localhost:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
