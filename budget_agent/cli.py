"""Command line interface.

    python -m budget_agent.cli demo                       # create a workspace with simulated data
    python -m budget_agent.cli load data/my_company.csv   # or load your own data
    python -m budget_agent.cli plan --budget 100000 --weeks 13 --min tv=15000
    python -m budget_agent.cli weekly --demo-weeks 4      # learning loop (schedule it weekly)
    python -m budget_agent.cli ask "I have $100K for Q1, how should I spend it?"
    python -m budget_agent.cli chat
    python -m budget_agent.cli evaluate
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .env import load_dotenv


def _kv(items: list[str] | None) -> dict[str, float]:
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k.strip()] = float(v)
    return out


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    ap = argparse.ArgumentParser(prog="budget_agent")
    ap.add_argument("--workspace", default=os.getenv("BUDGET_AGENT_WORKSPACE", "workspace"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="Load simulated company data (3 years + future stream).")
    d.add_argument("--seed", type=int, default=7)

    ld = sub.add_parser("load", help="Load your own weekly CSV.")
    ld.add_argument("csv")

    p = sub.add_parser("plan", help="Autonomous run: data check -> fit -> optimise -> alert -> plan file.")
    p.add_argument("--budget", type=float, required=True)
    p.add_argument("--weeks", type=int, default=13)
    p.add_argument("--min", nargs="*", help="channel=amount minimum total spend")
    p.add_argument("--max", nargs="*", help="channel=amount maximum total spend")
    p.add_argument("--max-change", type=float, help="max relative change per channel vs current (0.3 = 30%%)")

    w = sub.add_parser("weekly", help="Ingest new weeks, refit, re-plan, alert.")
    w.add_argument("--csv")
    w.add_argument("--demo-weeks", type=int)

    a = sub.add_parser("ask", help="One question to the LLM agent.")
    a.add_argument("question")
    sub.add_parser("chat", help="Interactive chat with the LLM agent.")

    e = sub.add_parser("evaluate", help="Accuracy, business value and ground-truth validation report.")
    e.add_argument("--budget", type=float, default=100000)

    args = ap.parse_args(argv)
    from .workspace import Workspace
    ws = Workspace(args.workspace)

    if args.cmd == "demo":
        print(json.dumps(ws.init_demo(seed=args.seed), indent=2))
    elif args.cmd == "load":
        from .data import load_csv
        print(json.dumps(ws.set_data(load_csv(args.csv), source=args.csv), indent=2))
    elif args.cmd == "plan":
        from .autopilot import run_goal
        run_goal(ws, args.budget, args.weeks, _kv(args.min) or None, _kv(args.max) or None, args.max_change)
    elif args.cmd == "weekly":
        from .autopilot import weekly_update
        weekly_update(ws, csv_path=args.csv, demo_weeks=args.demo_weeks)
    elif args.cmd in ("ask", "chat"):
        if not os.getenv("OPENAI_API_KEY"):
            print("OPENAI_API_KEY is not set (put it in .env). Use `plan` for the offline autopilot.")
            return 1
        from .agent import BudgetAgent
        agent = BudgetAgent(ws)

        def show(step):
            print(f"  -> {step.tool}({json.dumps(step.args)[:120]}) [{step.seconds:.1f}s]"
                  + (f" ERROR {step.result['error']}" if isinstance(step.result, dict) and "error" in step.result else ""))

        def turn(q):
            r = agent.ask(q, on_step=show)
            if r.plan:
                print("PLAN: " + " | ".join(r.plan[-1]["steps"]))
            print("\n" + r.answer)
            g = r.grounding
            print(f"\n[grounding: {len(g['grounded'])}/{g['numbers_checked']} figures verified"
                  + (f"; unverified: {g['ungrounded']}" if g["ungrounded"] else "") + "]")

        if args.cmd == "ask":
            turn(args.question)
        else:
            print("Budget Agent chat. Type 'exit' to quit.")
            while True:
                try:
                    q = input("\nyou> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if q.lower() in {"exit", "quit"}:
                    break
                if q:
                    turn(q)
    elif args.cmd == "evaluate":
        from .evaluation import business_value, ground_truth_recovery, holdout_accuracy
        from .mmm import MMM
        if ws.df is None:
            ws.init_demo()
        acc = holdout_accuracy(ws.df)
        acc.pop("series")
        model = ws.model or MMM().fit(ws.df)
        bv = business_value(model, args.budget, 13, df=ws.df)
        gt = ground_truth_recovery()
        report = {
            "1_accuracy_holdout": acc,
            "2_business_value": {k: v for k, v in bv.items() if k not in ("current", "optimized")},
            "3_validation_ground_truth": gt,
        }
        out = Path(args.workspace) / "evaluation_report.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str))
        print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
