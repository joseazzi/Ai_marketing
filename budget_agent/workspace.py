"""Persistent agent state + the actions (tools) the agent can take.

A workspace is a folder holding the current data, the fitted model, the agent's
memory (past recommendations and events), alerts and generated budget plans.
Because state persists, the agent can be re-run every week and *learn* from new
results: it refits, compares with what it recommended last time, and reports
what changed.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .mmm import MMM
from .notifier import read_alerts, send_alerts
from .optimizer import Plan, compare, current_allocation, evaluate_plan, explain_shifts, optimize, what_if

SATURATION_ALERT = 0.70     # share of max effect already reached
SHIFT_ALERT = 0.10          # recommendation moved by >10% of the channel's budget

DEMO_SHIFT_WEEK = 160
DEMO_SHIFT = {  # market change hidden in the demo stream: Instagram saturates, Google Ads gets stronger
    "instagram": {"half_sat": 1100, "beta": 5200},
    "google_ads": {"beta": 9500},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _usd(x: float) -> str:
    return f"-${-x:,.0f}" if x < 0 else f"${x:,.0f}"


def _r(x, n=2):
    return None if x is None else round(float(x), n)


class Workspace:
    def __init__(self, root: str | Path = "workspace", recency_halflife: float | None = 78, mmm_maxiter: int = 60,
                 gross_margin: float | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "plans").mkdir(exist_ok=True)
        self.recency_halflife = recency_halflife
        self.mmm_maxiter = mmm_maxiter
        self.df: pd.DataFrame | None = None
        self.model: MMM | None = None
        self.scenarios: dict[str, Plan] = {}
        self.last_goal: dict | None = None
        self.memory = self._load_memory()
        if (self.root / "data.csv").exists():
            self.df = D.load_csv(self.root / "data.csv")
        if (self.root / "model.json").exists():
            self.model = MMM.load(self.root / "model.json")
        self.last_goal = self.memory.get("last_goal")
        # gross margin: explicit argument > value saved from the UI/agent > GROSS_MARGIN in .env
        if gross_margin is None:
            if "gross_margin" in self.memory:
                gross_margin = self.memory["gross_margin"]
            elif os.getenv("GROSS_MARGIN"):
                gross_margin = float(os.getenv("GROSS_MARGIN"))
        self.gross_margin = gross_margin

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    @property
    def memory_path(self) -> Path:
        return self.root / "memory.json"

    @property
    def alerts_path(self) -> Path:
        return self.root / "alerts.jsonl"

    def _load_memory(self) -> dict:
        if self.memory_path.exists():
            return json.loads(self.memory_path.read_text())
        return {"recommendations": [], "events": [], "model_history": [], "last_goal": None}

    def _save_memory(self) -> None:
        self.memory["last_goal"] = self.last_goal
        self.memory_path.write_text(json.dumps(self.memory, indent=2, default=str))

    def log_event(self, kind: str, **details) -> None:
        self.memory["events"].append({"timestamp": _now(), "kind": kind, **details})
        self._save_memory()

    # ------------------------------------------------------------------ #
    # data
    # ------------------------------------------------------------------ #
    def set_data(self, df: pd.DataFrame, source: str = "upload") -> dict:
        self.df = D.clean(df)
        self.df.to_csv(self.root / "data.csv", index=False)
        self.model = None
        (self.root / "model.json").unlink(missing_ok=True)
        self.scenarios = {}
        self.memory["data_source"] = source
        self.memory.pop("holdout", None)
        self.log_event("data_loaded", source=source, weeks=len(self.df))
        return self.check_data()

    def init_demo(self, seed: int = 7, history_weeks: int = 156, stream_weeks: int = 26) -> dict:
        """Simulated company: 3 years of history + a stream of future weeks (with a hidden market shift)."""
        sim = D.simulate(weeks=history_weeks + stream_weeks, seed=seed,
                         shift_week=DEMO_SHIFT_WEEK, shift=DEMO_SHIFT)
        sim.data.iloc[history_weeks:].to_csv(self.root / "demo_stream.csv", index=False)
        return self.set_data(sim.data.iloc[:history_weeks], source=f"simulated (seed={seed})")

    def demo_weeks_remaining(self) -> int:
        p = self.root / "demo_stream.csv"
        if not p.exists() or self.df is None:
            return 0
        s = D.load_csv(p)
        return int((s[D.DATE_COL] > self.df[D.DATE_COL].max()).sum())

    def next_demo_weeks(self, n: int = 4) -> pd.DataFrame:
        s = D.load_csv(self.root / "demo_stream.csv")
        return s[s[D.DATE_COL] > self.df[D.DATE_COL].max()].head(n)

    @property
    def breakeven(self) -> float:
        """Sales needed from $1 of ads to break even: 1.0 on revenue, 1/margin on profit."""
        return 1.0 / self.gross_margin if self.gross_margin else 1.0

    def _profit(self, plan: Plan) -> float | None:
        return plan.total_sales * self.gross_margin - plan.total_budget if self.gross_margin else None

    def _require_data(self):
        if self.df is None:
            raise RuntimeError("No data loaded. Upload a CSV or load the demo dataset first.")

    def _require_model(self):
        self._require_data()
        if self.model is None:
            raise RuntimeError("No model trained yet. Call fit_model first.")

    # ================================================================== #
    # TOOLS (each returns a JSON-serialisable dict)
    # ================================================================== #
    def set_gross_margin(self, gross_margin: float | None) -> dict:
        """Set the gross margin (0-1) so recommendations are judged on profit, not just revenue."""
        if gross_margin is not None and not 0 < gross_margin <= 1:
            return {"error": "gross_margin must be between 0 and 1 (e.g. 0.55 for 55%)."}
        self.gross_margin = gross_margin
        self.memory["gross_margin"] = gross_margin  # survives restarts, like memory and plans
        self.log_event("gross_margin_set", gross_margin=gross_margin)
        return {"gross_margin": gross_margin, "breakeven_sales_per_ad_dollar": _r(self.breakeven, 3),
                "meaning": f"A channel pays for itself only if $1 of ads brings at least ${self.breakeven:.2f} of sales."}

    @property
    def data_source(self) -> str:
        return self.memory.get("data_source", "unknown")

    @property
    def is_simulated(self) -> bool:
        return self.data_source.startswith("simulated")

    def check_data(self) -> dict:
        self._require_data()
        return {**D.quality_report(self.df), "data_source": self.data_source,
                "is_simulated_demo": self.is_simulated}

    def _holdout(self, halflife) -> dict | None:
        """Accuracy on the last 13 weeks when the model is trained without them (cached per dataset)."""
        from .evaluation import holdout_accuracy
        key = {"weeks": len(self.df), "data_end": str(self.df[D.DATE_COL].max().date()), "halflife": halflife}
        cached = self.memory.get("holdout")
        if cached and cached.get("key") == key:
            return cached["result"]
        if len(self.df) < 65:  # too short to hide 13 weeks and still train
            return None
        h = holdout_accuracy(self.df, test_weeks=13, recency_halflife=halflife or None, maxiter=self.mmm_maxiter)
        result = {"test_weeks": 13, "holdout_r2": _r(h["test_r2"], 3), "holdout_mape": _r(h["test_mape"], 4),
                  "meaning": "accuracy on the last 13 weeks when the model is trained without them"}
        self.memory["holdout"] = {"key": key, "result": result}
        self._save_memory()
        return result

    def fit_model(self, recency_halflife_weeks: float | None = None, from_scratch: bool = False) -> dict:
        self._require_data()
        q = D.quality_report(self.df)
        if not q["ok_to_model"]:
            return {"error": "Data quality issues block modelling.", "issues": q["issues"]}
        hl = recency_halflife_weeks if recency_halflife_weeks is not None else self.recency_halflife
        previous = self.model
        use_prior = previous is not None and not from_scratch and previous.trained_weeks <= len(self.df)
        self.model = MMM(recency_halflife=hl or None, maxiter=self.mmm_maxiter).fit(
            self.df, prior=previous if use_prior else None)
        self.model.save(self.root / "model.json")
        summ = self.model.channel_summary(self.df)
        entry = {
            "timestamp": _now(), "data_end": str(self.df[D.DATE_COL].max().date()), "weeks": len(self.df),
            "mode": "update" if use_prior else "scratch",
            "r2": _r(self.model.fit_metrics["r2"], 4), "mape": _r(self.model.fit_metrics["mape"], 4),
            "roi": {r.channel: _r(r.roi, 3) for r in summ.itertuples()},
        }
        self.memory["model_history"].append(entry)
        self._save_memory()
        out = {
            "trained_on_weeks": len(self.df),
            "r2": entry["r2"], "mape": entry["mape"],
            "fit_quality": "good" if entry["r2"] > 0.8 and entry["mape"] < 0.1 else "weak - treat recommendations with caution",
            "recency_halflife_weeks": hl,
            "mode": "updated previous model with new evidence" if use_prior else "trained from scratch",
            "roi_by_channel": entry["roi"],
            "data_period": {"start": str(self.df[D.DATE_COL].min().date()), "end": entry["data_end"],
                            "weeks": len(self.df)},
            "data_source": self.data_source,
            "is_simulated_demo": self.is_simulated,
            "holdout_accuracy": self._holdout(hl),
            "how_effects_are_estimated": "Channel effects are estimated from past week-to-week variation in spend "
                                         "and sales, not measured by experiments. Channels whose spend moves "
                                         "together are hard to separate.",
            "channels_hard_to_separate": [w for w in q["warnings"] if "correlated" in w],
        }
        if previous is not None and len(self.memory["model_history"]) >= 2:
            prev = self.memory["model_history"][-2]["roi"]
            out["roi_change_vs_previous_model"] = {c: _r(entry["roi"][c] - prev.get(c, 0), 3) for c in entry["roi"]}
        return out

    def get_channel_performance(self) -> dict:
        self._require_model()
        s = self.model.channel_summary(self.df, breakeven=self.breakeven)
        rows = []
        for r in s.itertuples():
            rows.append({
                "channel": r.channel, "roi": _r(r.roi, 3), "marginal_roi_now": _r(r.marginal_roi, 3),
                "share_of_spend": _r(r.share_of_spend, 3), "share_of_media_sales": _r(r.share_of_media_sales, 3),
                "current_weekly_spend": _r(r.current_weekly_spend), "saturation_pct": _r(r.saturation_pct, 3),
                "weekly_spend_where_next_dollar_stops_paying_off": _r(r.saturation_weekly_spend),
                "ad_effect_lasts_weeks": _r(r.carryover_weeks, 1),
                "status": ("saturated" if r.saturation_pct > SATURATION_ALERT
                           else "below break-even" if r.marginal_roi < self.breakeven
                           else "room to grow"),
            })
            if self.gross_margin:
                rows[-1]["profit_per_ad_dollar"] = _r(r.roi * self.gross_margin - 1, 3)
        dec = self.model.decompose(self.df)
        media = float(dec[self.model.channels].sum().sum())
        return {
            "channels": rows,
            "best_channel_by_roi": rows[0]["channel"], "worst_channel_by_roi": rows[-1]["channel"],
            "media_driven_share_of_sales": _r(media / float(self.df[D.TARGET_COL].sum()), 3),
            "gross_margin": self.gross_margin,
            "breakeven_sales_per_ad_dollar": _r(self.breakeven, 3),
            "note": "Historical view. roi = sales per $ over the whole history; marginal_roi_now = sales from the NEXT $ "
                    "at the channel's RECENT HISTORICAL weekly spend (last 13 weeks), not at any plan's spend. "
                    "To explain a budget plan, use optimize_budget's marginal_roi fields instead."
                    + ("" if self.gross_margin else " No gross margin set: results are revenue, not profit."),
        }

    def optimize_budget(self, total_budget: float, n_weeks: int = 13, min_spend: dict | None = None,
                        max_spend: dict | None = None, max_change_pct: float | None = None,
                        goal: str = "max_sales") -> dict:
        """goal="max_sales" spends the whole budget; goal="max_profit" (needs a gross margin) treats the budget
        as a cap and stops spending where the next dollar no longer pays for itself."""
        self._require_model()
        if goal == "max_profit" and not self.gross_margin:
            return {"error": "goal=max_profit needs a gross margin. Call set_gross_margin first (ask the user)."}
        self.last_goal = {"total_budget": total_budget, "n_weeks": n_weeks, "min_spend": min_spend or {},
                          "max_spend": max_spend or {}, "max_change_pct": max_change_pct, "goal": goal}
        cur = evaluate_plan(self.model, current_allocation(self.model, total_budget, df=self.df), n_weeks, "current split")
        best = optimize(self.model, total_budget, n_weeks, min_spend=min_spend, max_spend=max_spend,
                        max_change_vs=cur.allocation if max_change_pct is not None else None,
                        max_change_pct=max_change_pct,
                        goal="min_budget_for_roi" if goal == "max_profit" else "max_sales",
                        target_roi=self.breakeven if goal == "max_profit" else None)
        best.name = "recommended"
        profit_note = None
        if self.gross_margin and goal == "max_sales":
            profit_note = self._profit_note(best, total_budget, n_weeks, min_spend, max_spend, max_change_pct, cur)
        self.scenarios["current split"] = cur
        self.scenarios["recommended"] = best
        start = (self.df[D.DATE_COL].max() + pd.Timedelta(days=7)).date()
        baseline = self.model.baseline_forecast(start, n_weeks)
        rec = {
            "timestamp": _now(), "data_end": str(self.df[D.DATE_COL].max().date()), **self.last_goal,
            "allocation": {c: _r(v) for c, v in best.allocation.items()},
            "expected_incremental_sales": _r(best.total_sales),
        }
        self.memory["recommendations"].append(rec)
        self._save_memory()
        return {
            "budget": total_budget, "n_weeks": n_weeks, "plan_starts": str(start),
            "recommended": best.to_dict(), "current_split": cur.to_dict(),
            "uplift_vs_current": _r(best.total_sales - cur.total_sales),
            "uplift_vs_current_pct": _r(best.total_sales / cur.total_sales - 1, 4) if cur.total_sales else None,
            "expected_total_sales_incl_baseline": _r(baseline + best.total_sales),
            "biggest_moves": explain_shifts(self.model, cur, best,
                                            breakeven=self.breakeven if goal == "max_profit" else 1.0),
            "constraints_applied": {k: v for k, v in self.last_goal.items() if v},
            **({"change_limit_rule": "In profit mode max_change_pct only caps increases; cuts are always allowed so "
                                     "no channel is forced to lose money. Use goal='max_sales' to keep every channel "
                                     "within +/-max_change_pct."}
               if goal == "max_profit" and max_change_pct is not None else {}),
            **({"profit_note": profit_note} if profit_note else {}),
            "marginal_roi_basis": "current_split.marginal_roi and recommended.marginal_roi are the sales from the next $ "
                                  "at each plan's own weekly spend for this budget (use these to explain the plan).",
            **({"gross_margin": self.gross_margin,
                "recommended_profit": _r(self._profit(best)), "current_split_profit": _r(self._profit(cur)),
                "profit_uplift_vs_current": _r(self._profit(best) - self._profit(cur)),
                "unspent_budget": _r(total_budget - best.total_budget)} if self.gross_margin else
               {"note": "No gross margin set: figures are revenue. Ask for the margin to judge profit."}),
        }

    def _profit_note(self, plan: Plan, budget, n_weeks, min_spend, max_spend, max_change_pct, cur) -> str | None:
        """Warn when a sales-maximising plan spends past break-even, and give the profit alternative."""
        funded = [c for c, v in plan.allocation.items() if v > 1]
        if not funded:
            return None
        rets = [plan.marginal_roi[c] for c in funded]
        lo, hi = min(rets), max(rets)
        if lo >= self.breakeven:
            return None
        ret_txt = f"${lo:.2f}" if hi - lo < 0.01 else f"between ${lo:.2f} and ${hi:.2f}"
        note = (f"At the plan's spend the next $1 returns {ret_txt} in sales, below break-even "
                f"${self.breakeven:.2f} at a {self.gross_margin:.0%} gross margin: the last dollars lose money. "
                f"This plan's profit from ads is ${self._profit(plan):,.0f}.")
        try:
            alt = optimize(self.model, budget, n_weeks, min_spend=min_spend, max_spend=max_spend,
                           max_change_vs=cur.allocation if max_change_pct is not None else None,
                           max_change_pct=max_change_pct, goal="min_budget_for_roi", target_roi=self.breakeven)
            note += (f" goal='max_profit' would give ${self._profit(alt):,.0f} profit spending "
                     f"${alt.total_budget:,.0f} of the ${budget:,.0f}.")
        except (ValueError, RuntimeError):
            note += " A profit-maximising plan is not possible with these limits."
        return note

    def run_what_if(self, changes: dict, base: str = "current split", reallocate: bool = False,
                    total_budget: float | None = None, n_weeks: int | None = None) -> dict:
        self._require_model()
        if base not in self.scenarios:
            tb = total_budget or (self.last_goal or {}).get("total_budget")
            nw = n_weeks or (self.last_goal or {}).get("n_weeks", 13)
            if not tb:
                raise RuntimeError("Give total_budget (or run optimize_budget first) so there is a base plan.")
            self.scenarios["current split"] = evaluate_plan(
                self.model, current_allocation(self.model, tb, df=self.df), nw, "current split")
            base = "current split" if base not in self.scenarios else base
        b = self.scenarios[base]
        p = what_if(self.model, b.allocation, b.n_weeks, {k: float(v) for k, v in changes.items()}, reallocate)
        self.scenarios[p.name] = p
        extra = ({"profit": _r(self._profit(p)), "profit_change_vs_base": _r(self._profit(p) - self._profit(b))}
                 if self.gross_margin else {})
        return {
            "scenario": p.name, "base": base, "result": p.to_dict(), **extra,
            "sales_change_vs_base": _r(p.total_sales - b.total_sales),
            "sales_change_vs_base_pct": _r(p.total_sales / b.total_sales - 1, 4) if b.total_sales else None,
            "budget_change_vs_base": _r(p.total_budget - b.total_budget),
        }

    def compare_scenarios(self, names: list[str] | None = None) -> dict:
        self._require_model()
        names = names or list(self.scenarios)
        missing = [n for n in names if n not in self.scenarios]
        if missing:
            return {"error": f"Unknown scenarios {missing}", "available": list(self.scenarios)}
        rows = compare([self.scenarios[n] for n in names])
        best = max(rows, key=lambda r: r["incremental_sales"])
        return {"scenarios": rows, "best_by_sales": best["name"], "best_by_roi": max(rows, key=lambda r: r["roi"])["name"]}

    def check_saturation_alerts(self, notify: bool = True) -> dict:
        """Find channels where extra money is being wasted (or opportunities) and push alerts."""
        self._require_model()
        perf = self.get_channel_performance()["channels"]
        alerts = []
        for r in perf:
            if r["saturation_pct"] is not None and r["saturation_pct"] > SATURATION_ALERT:
                alerts.append({"severity": "warning", "channel": r["channel"], "type": "saturation",
                               "message": f"{r['channel']} is {r['saturation_pct']:.0%} saturated at "
                                          f"${r['current_weekly_spend']:,.0f}/week; extra spend there returns "
                                          f"${r['marginal_roi_now']:.2f} per $1."})
            elif r["marginal_roi_now"] is not None and r["marginal_roi_now"] < self.breakeven:
                why = (f" (break-even at a {self.gross_margin:.0%} margin is ${self.breakeven:.2f})"
                       if self.gross_margin else "")
                alerts.append({"severity": "warning", "channel": r["channel"], "type": "low_marginal_return",
                               "message": f"The next $1 on {r['channel']} returns only ${r['marginal_roi_now']:.2f} "
                                          f"in sales{why}."})
            elif r["marginal_roi_now"] and r["marginal_roi_now"] > 2.0 * self.breakeven:
                alerts.append({"severity": "info", "channel": r["channel"], "type": "opportunity",
                               "message": f"{r['channel']} has room to grow: the next $1 returns "
                                          f"${r['marginal_roi_now']:.2f}."})
        delivered = send_alerts(alerts, self.alerts_path) if notify else []
        return {"alerts": alerts, "delivered_via": delivered}

    def generate_budget_plan(self, scenario: str = "recommended", start_date: str | None = None) -> dict:
        """Write a ready-to-use plan: weekly spend per channel (CSV) + a one-page brief (Markdown)."""
        self._require_model()
        if scenario not in self.scenarios:
            return {"error": f"Scenario '{scenario}' not found. Run optimize_budget first.",
                    "available": list(self.scenarios)}
        p = self.scenarios[scenario]
        start = pd.Timestamp(start_date) if start_date else self.df[D.DATE_COL].max() + pd.Timedelta(days=7)
        weeks = pd.date_range(start, periods=p.n_weeks, freq="7D")
        flight = pd.DataFrame({"week_start": weeks.date})
        for ch in self.model.channels:
            flight[ch] = np.round(p.allocation[ch] / p.n_weeks, 2)
        flight["total"] = flight[self.model.channels].sum(axis=1).round(2)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        slug = "".join(c if c.isalnum() else "_" for c in scenario)[:40]
        csv_path = self.root / "plans" / f"budget_plan_{slug}_{stamp}.csv"
        md_path = csv_path.with_suffix(".md")
        flight.to_csv(csv_path, index=False)

        cur = self.scenarios.get("current split")
        goal = (self.last_goal or {}).get("goal", "max_sales") if scenario == "recommended" else "max_sales"
        profit_plan = goal == "max_profit" and self.gross_margin
        cap = (self.last_goal or {}).get("total_budget", p.total_budget)
        lines = [
            f"# Budget plan: {scenario}" + (" (maximise profit)" if profit_plan else ""),
            "",
            f"*Period:* {weeks[0].date()} to {(weeks[-1] + pd.Timedelta(days=6)).date()} ({p.n_weeks} weeks)  ",
        ]
        if profit_plan and cap - p.total_budget > 1:
            lines.append(f"*Budget:* spend ${p.total_budget:,.0f} of ${cap:,.0f}. Keep ${cap - p.total_budget:,.0f} "
                         f"unspent, because spending it would lower profit  ")
        else:
            lines.append(f"*Total budget:* ${p.total_budget:,.0f}  ")
        lines.append(f"*Expected incremental sales from media:* ${p.total_sales:,.0f} (ROI {p.roi:.2f})  ")
        if self.gross_margin:
            lines.append(f"*Expected profit from ads* (at a {self.gross_margin:.0%} gross margin, break-even "
                         f"${self.breakeven:.2f} of sales per $1): {_usd(self._profit(p))}  ")
        if cur and scenario != "current split":
            if self.gross_margin:
                lines.append(f"*Versus current split:* {self._profit(p) - self._profit(cur):+,.0f} profit "
                             f"(current split: {_usd(self._profit(cur))}); "
                             f"{p.total_sales - cur.total_sales:+,.0f} sales ({p.total_sales / cur.total_sales - 1:+.1%})")
            else:
                lines.append(f"*Versus current split:* {p.total_sales - cur.total_sales:+,.0f} sales "
                             f"({p.total_sales / cur.total_sales - 1:+.1%})")
        lines += ["", "| Channel | Period budget | Weekly spend | Expected sales | Next-$ return | Saturation |",
                  "|---|---:|---:|---:|---:|---:|"]
        for ch in sorted(self.model.channels, key=lambda c: -p.allocation[c]):
            lines.append(f"| {ch} | ${p.allocation[ch]:,.0f} | ${p.allocation[ch] / p.n_weeks:,.0f} | "
                         f"${p.expected_sales[ch]:,.0f} | ${p.marginal_roi[ch]:.2f} | {p.saturation_pct[ch]:.0%} |")
        if cur and scenario != "current split":
            lines += ["", "## Why these changes"]
            for m in explain_shifts(self.model, cur, p, breakeven=self.breakeven if profit_plan else 1.0):
                lines.append(f"- **{m['channel']}** {m['change']:+,.0f}: {m['reason']} "
                             f"(next-$ return {m['current_marginal_roi']:.2f} -> {m['optimized_marginal_roi']:.2f}).")
        lines += ["", f"_Generated by the Budget Agent on {_now()} from data up to "
                      f"{self.df[D.DATE_COL].max().date()} (model R² {self.model.fit_metrics['r2']:.2f})._"]
        md_path.write_text("\n".join(lines))
        self.log_event("plan_generated", scenario=scenario, csv=str(csv_path), brief=str(md_path))
        return {"csv_path": str(csv_path), "brief_path": str(md_path), "weeks": p.n_weeks,
                "weekly_spend": {c: _r(p.allocation[c] / p.n_weeks) for c in self.model.channels}}

    def ingest_new_data(self, new_rows: pd.DataFrame | None = None, csv_path: str | None = None,
                        demo_weeks: int | None = None) -> dict:
        """Learning loop: add new weekly results, refit, re-optimize the last goal, report what changed."""
        self._require_data()
        if new_rows is None:
            if csv_path:
                new_rows = D.load_csv(csv_path)
            elif demo_weeks:
                new_rows = self.next_demo_weeks(demo_weeks)
            else:
                raise RuntimeError("Provide new weekly rows, a csv_path, or demo_weeks.")
        new_rows = D.clean(new_rows)
        if new_rows.empty:
            return {"error": "No new weeks to ingest."}
        before_end = self.df[D.DATE_COL].max()
        old_rec = self.memory["recommendations"][-1] if self.memory["recommendations"] else None
        old_perf = {r["channel"]: r for r in self.get_channel_performance()["channels"]} if self.model else {}

        merged = pd.concat([self.df, new_rows]).drop_duplicates(subset=D.DATE_COL, keep="last")
        self.df = D.clean(merged)
        self.df.to_csv(self.root / "data.csv", index=False)
        added = int((self.df[D.DATE_COL] > before_end).sum())

        # how well did the previous model predict the new weeks? (honest out-of-sample check)
        oos = None
        if self.model is not None:
            pred = self.model.predict(self.df)[-added:] if added else []
            act = self.df[D.TARGET_COL].to_numpy()[-added:] if added else []
            if added:
                oos = {"weeks": added, "mape": _r(np.mean(np.abs((act - pred) / act)), 4),
                       "actual_sales": _r(act.sum()), "predicted_sales": _r(np.sum(pred))}

        fit = self.fit_model()
        out = {"weeks_added": added, "data_now_ends": str(self.df[D.DATE_COL].max().date()),
               "previous_model_out_of_sample": oos, "refit": fit}
        if "error" in fit or self.model is None:
            out["warning"] = "New weeks were saved, but the model could not be retrained: " + fit.get("error", "")
            self.log_event("new_data_ingested", weeks_added=added, alerts=0, refit_failed=True)
            return out

        new_perf = {r["channel"]: r for r in self.get_channel_performance()["channels"]}
        out["channel_changes"] = {
            c: {"roi_before": old_perf[c]["roi"], "roi_after": new_perf[c]["roi"],
                "marginal_roi_before": old_perf[c]["marginal_roi_now"], "marginal_roi_after": new_perf[c]["marginal_roi_now"],
                "saturation_before": old_perf[c]["saturation_pct"], "saturation_after": new_perf[c]["saturation_pct"]}
            for c in new_perf if c in old_perf
        }

        alerts = []
        res = None
        if self.last_goal:
            g = dict(self.last_goal)
            wanted = g.get("goal", "max_sales")
            attempts = [dict(goal=wanted)] + ([dict(goal="max_sales")] if wanted != "max_sales" else [])
            for extra in attempts:  # the old goal can become impossible (e.g. margin removed, limits infeasible)
                try:
                    res = self.optimize_budget(g["total_budget"], g["n_weeks"], g.get("min_spend") or None,
                                               g.get("max_spend") or None, g.get("max_change_pct"), **extra)
                except (ValueError, RuntimeError) as e:
                    res = {"error": str(e)}
                if "error" not in res:
                    if extra["goal"] != wanted:
                        out["goal_fallback"] = (f"The '{wanted}' goal was not possible this week, so this plan "
                                                f"maximises sales. Your goal stays '{wanted}' for next weeks.")
                    break
            self.last_goal = g  # keep the user's original goal (a one-week fallback must not replace it)
            self._save_memory()
            if "error" in res:
                out["reoptimize_error"] = (f"The model was updated, but the last plan could not be re-optimized: "
                                           f"{res['error']}. Run optimize_budget with new limits.")
                self.last_goal = g
                res = None
        if res is not None:
            g = self.last_goal
            new_alloc = res["recommended"]["allocation"]
            out["new_recommendation"] = {"allocation": new_alloc,
                                         "expected_incremental_sales": res["recommended"]["total_incremental_sales"]}
            if old_rec and abs(old_rec["total_budget"] - g["total_budget"]) < 1e-6:
                shifts = {c: _r(new_alloc[c] - old_rec["allocation"].get(c, 0)) for c in new_alloc}
                out["recommendation_shift"] = shifts
                for c, d in shifts.items():
                    if abs(d) > SHIFT_ALERT * max(old_rec["allocation"].get(c, 0), g["total_budget"] / len(shifts)):
                        alerts.append({"severity": "warning", "channel": c, "type": "recommendation_changed",
                                       "message": f"New data changed the plan for {c}: {d:+,.0f} "
                                                  f"(now ${new_alloc[c]:,.0f} for {g['n_weeks']} weeks)."})
        for c, ch in out["channel_changes"].items():
            if (ch["saturation_before"] or 0) <= SATURATION_ALERT < (ch["saturation_after"] or 0):
                alerts.append({"severity": "warning", "channel": c, "type": "became_saturated",
                               "message": f"{c} just crossed {SATURATION_ALERT:.0%} saturation."})
        out["alerts"] = alerts
        out["delivered_via"] = send_alerts(alerts, self.alerts_path)
        self.log_event("new_data_ingested", weeks_added=added, alerts=len(alerts))
        return out

    def get_history(self, limit: int = 5) -> dict:
        return {
            "recent_recommendations": self.memory["recommendations"][-limit:],
            "model_history": self.memory["model_history"][-limit:],
            "recent_events": self.memory["events"][-limit:],
            "recent_alerts": read_alerts(self.alerts_path, limit),
        }
