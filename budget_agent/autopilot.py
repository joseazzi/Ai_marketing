"""Deterministic agent workflows that need no LLM.

* ``run_goal``     - plan -> check data -> fit -> analyse -> optimise -> alert -> write plan -> explain.
* ``weekly_update`` - the learning loop, designed to run on a schedule (cron / GitHub Actions).

Used as an offline fallback when no OpenAI key is set, for tests, and for scheduled runs.
"""
from __future__ import annotations

from .workspace import Workspace


def run_goal(ws: Workspace, total_budget: float, n_weeks: int = 13, min_spend=None, max_spend=None,
             max_change_pct=None, log=print) -> dict:
    plan = ["check_data", "fit_model", "get_channel_performance", "optimize_budget",
            "check_saturation_alerts", "generate_budget_plan"]
    log(f"PLAN for ${total_budget:,.0f} over {n_weeks} weeks: " + " -> ".join(plan))
    trace = {}

    q = ws.check_data()
    trace["check_data"] = q
    log(f"[1] Data: {q['weeks']} weeks ({q['start']} to {q['end']}), {len(q['channels'])} channels. "
        f"Issues: {q['issues'] or 'none'}. Warnings: {len(q['warnings'])}.")
    if not q["ok_to_model"]:
        log("Stopping: data quality issues must be fixed first.")
        return {"plan": plan, "trace": trace, "summary": "Blocked by data quality issues: " + "; ".join(q["issues"])}

    if ws.model is None or ws.model.trained_weeks != len(ws.df):
        f = ws.fit_model()
        trace["fit_model"] = f
        log(f"[2] Model trained: R² {f['r2']:.3f}, MAPE {f['mape']:.1%} ({f['fit_quality']}).")
    else:
        log("[2] Model is up to date with the data; skipping refit.")

    perf = ws.get_channel_performance()
    trace["get_channel_performance"] = perf
    log("[3] ROI by channel: " + ", ".join(f"{c['channel']} {c['roi']:.2f}" for c in perf["channels"]))

    opt = ws.optimize_budget(total_budget, n_weeks, min_spend, max_spend, max_change_pct)
    trace["optimize_budget"] = opt
    log(f"[4] Optimized: {opt['recommended']['total_incremental_sales']:,.0f} incremental sales vs "
        f"{opt['current_split']['total_incremental_sales']:,.0f} with the current split "
        f"({opt['uplift_vs_current_pct']:+.1%}).")

    al = ws.check_saturation_alerts(notify=True)
    trace["check_saturation_alerts"] = al
    log(f"[5] Alerts: {len(al['alerts'])} sent via {al['delivered_via'] or ['none']}.")

    gp = ws.generate_budget_plan("recommended")
    trace["generate_budget_plan"] = gp
    log(f"[6] Plan written: {gp['csv_path']} and {gp['brief_path']}")

    summary = explain(opt)
    log("\n" + summary)
    return {"plan": plan, "trace": trace, "summary": summary}


def explain(opt: dict) -> str:
    rec, cur = opt["recommended"], opt["current_split"]
    lines = [f"Recommendation for ${opt['budget']:,.0f} over {opt['n_weeks']} weeks "
             f"(starting {opt['plan_starts']}):", ""]
    for ch, v in sorted(rec["allocation"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {ch:<12} ${v:>10,.0f}   (now ${cur['allocation'][ch]:>10,.0f})")
    lines += ["", f"Expected incremental sales: ${rec['total_incremental_sales']:,.0f} vs "
                  f"${cur['total_incremental_sales']:,.0f} today ({opt['uplift_vs_current_pct']:+.1%}).", "Why:"]
    for m in opt["biggest_moves"]:
        verb = "Increase" if m["change"] > 0 else "Reduce"
        lines.append(f"  - {verb} {m['channel']} by ${abs(m['change']):,.0f}: {m['reason']} "
                     f"(next-$ return {m['current_marginal_roi']:.2f}, saturation {m['current_saturation_pct']:.0%}).")
    return "\n".join(lines)


def weekly_update(ws: Workspace, csv_path: str | None = None, demo_weeks: int | None = None, log=print) -> dict:
    res = ws.ingest_new_data(csv_path=csv_path, demo_weeks=demo_weeks)
    if "error" in res:
        log(res["error"])
        return res
    log(f"Ingested {res['weeks_added']} new weeks; data now ends {res['data_now_ends']}.")
    if res["previous_model_out_of_sample"]:
        o = res["previous_model_out_of_sample"]
        log(f"Last model's forecast error on the new weeks: MAPE {o['mape']:.1%}.")
    log(f"Refit: R² {res['refit']['r2']:.3f}.")
    for c, ch in res["channel_changes"].items():
        if abs(ch["roi_after"] - ch["roi_before"]) > 0.05:
            log(f"  {c}: ROI {ch['roi_before']:.2f} -> {ch['roi_after']:.2f}, "
                f"next-$ return {ch['marginal_roi_before']:.2f} -> {ch['marginal_roi_after']:.2f}")
    if "recommendation_shift" in res:
        log("Recommendation changes: " + ", ".join(f"{c} {d:+,.0f}" for c, d in res["recommendation_shift"].items()))
    log(f"Alerts: {len(res['alerts'])} ({', '.join(res['delivered_via']) or 'none'}).")
    if ws.last_goal:
        gp = ws.generate_budget_plan("recommended")
        res["plan_files"] = gp
        log(f"Updated plan written: {gp['csv_path']}")
    return res
