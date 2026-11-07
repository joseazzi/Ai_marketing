"""Tool schemas exposed to the LLM, and the dispatcher that runs them on a Workspace."""
from __future__ import annotations

import json
import traceback

from .workspace import Workspace

_CHANNEL_MAP = {"type": "object", "additionalProperties": {"type": "number"},
                "description": "Map of channel name -> amount, e.g. {\"tv\": 20000}"}

TOOLS = [
    {
        "name": "make_plan",
        "description": "Write down the step-by-step plan you will follow for this request BEFORE using other tools. "
                       "Call it first for every new goal; call again if the plan changes.",
        "parameters": {"type": "object", "properties": {
            "goal": {"type": "string", "description": "The user's goal in one sentence."},
            "steps": {"type": "array", "items": {"type": "string"}, "description": "Ordered steps (tool names + purpose)."},
        }, "required": ["goal", "steps"]},
    },
    {
        "name": "check_data",
        "description": "Data-quality report: weeks, date range, channels, controls, issues and warnings. Run before modelling.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "fit_model",
        "description": "Train the Media Mix Model (adstock + saturation) on the current data. Returns R², MAPE and ROI by channel.",
        "parameters": {"type": "object", "properties": {
            "recency_halflife_weeks": {"type": "number", "description": "Weight recent weeks more (half-life in weeks). Omit for the default."},
            "from_scratch": {"type": "boolean", "description": "Ignore the previous model (default: update it with the new evidence)."},
        }},
    },
    {
        "name": "get_channel_performance",
        "description": "Per-channel ROI, marginal ROI of the next dollar, saturation level, carry-over and status.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "optimize_budget",
        "description": "Find the budget split that maximises incremental sales for a total budget over n_weeks, "
                       "compared against the current split. Stores scenarios 'recommended' and 'current split'.",
        "parameters": {"type": "object", "properties": {
            "total_budget": {"type": "number"},
            "n_weeks": {"type": "integer", "description": "Planning period in weeks (a quarter = 13).", "default": 13},
            "min_spend": {**_CHANNEL_MAP, "description": "Minimum TOTAL spend per channel for the period."},
            "max_spend": {**_CHANNEL_MAP, "description": "Maximum TOTAL spend per channel for the period."},
            "max_change_pct": {"type": "number", "description": "Limit each channel to +/- this fraction of its current budget (0.3 = 30%)."},
        }, "required": ["total_budget"]},
    },
    {
        "name": "run_what_if",
        "description": "Test a scenario: apply relative changes to a base plan, e.g. {\"tv\": -0.3} = cut TV by 30%.",
        "parameters": {"type": "object", "properties": {
            "changes": {"type": "object", "additionalProperties": {"type": "number"},
                        "description": "channel -> relative change (-0.3 = -30%, 0.5 = +50%)."},
            "base": {"type": "string", "description": "Base scenario name: 'current split' (default) or 'recommended'."},
            "reallocate": {"type": "boolean", "description": "Move freed money to other channels (keep total constant)."},
            "total_budget": {"type": "number", "description": "Only needed if no plan exists yet."},
            "n_weeks": {"type": "integer"},
        }, "required": ["changes"]},
    },
    {
        "name": "compare_scenarios",
        "description": "Side-by-side comparison of stored scenarios (budget, incremental sales, ROI, allocation).",
        "parameters": {"type": "object", "properties": {
            "names": {"type": "array", "items": {"type": "string"}, "description": "Scenario names; omit for all."},
        }},
    },
    {
        "name": "check_saturation_alerts",
        "description": "Detect saturated channels / wasted spend / opportunities and SEND alerts (log, Slack, webhook, email).",
        "parameters": {"type": "object", "properties": {"notify": {"type": "boolean", "default": True}}},
    },
    {
        "name": "generate_budget_plan",
        "description": "ACTION: write a ready-to-use budget plan file (weekly spend per channel CSV + one-page brief).",
        "parameters": {"type": "object", "properties": {
            "scenario": {"type": "string", "default": "recommended"},
            "start_date": {"type": "string", "description": "YYYY-MM-DD; default = week after the data ends."},
        }},
    },
    {
        "name": "ingest_new_data",
        "description": "LEARN: add new weekly results (csv_path, or demo_weeks from the demo stream), refit, "
                       "re-optimise the last goal and report what changed.",
        "parameters": {"type": "object", "properties": {
            "csv_path": {"type": "string"},
            "demo_weeks": {"type": "integer", "description": "Number of new demo weeks to ingest."},
        }},
    },
    {
        "name": "get_history",
        "description": "Agent memory: past recommendations, model versions, events and alerts.",
        "parameters": {"type": "object", "properties": {"limit": {"type": "integer", "default": 5}}},
    },
]


def openai_tools() -> list[dict]:
    return [{"type": "function", "function": t} for t in TOOLS]


def dispatch(ws: Workspace, name: str, args: dict, plan_sink: list | None = None) -> dict:
    """Run a tool by name. Errors are returned as data so the agent can recover."""
    try:
        if name == "make_plan":
            if plan_sink is not None:
                plan_sink.append({"goal": args.get("goal"), "steps": args.get("steps", [])})
            return {"ok": True, "message": "Plan recorded. Execute it step by step."}
        fn = getattr(ws, name, None)
        if fn is None or name not in {t["name"] for t in TOOLS}:
            return {"error": f"Unknown tool {name}"}
        return fn(**args)
    except Exception as e:  # surface the error to the model instead of crashing
        return {"error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc(limit=2)}


def to_json(obj) -> str:
    return json.dumps(obj, default=str)
