"""The three evaluation pillars from the proposal: accuracy, business value, validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SimulationResult, simulate, true_roi
from .mmm import MMM, mape, r2
from .optimizer import current_allocation, evaluate_plan, optimize


def holdout_accuracy(df: pd.DataFrame, test_weeks: int = 13, **mmm_kwargs) -> dict:
    """1) Accuracy: train on all but the last ``test_weeks``, predict them."""
    train, test = df.iloc[:-test_weeks], df.iloc[-test_weeks:]
    m = MMM(**mmm_kwargs).fit(train)
    # predict with full history so adstock carry-over into the test period is correct
    pred_all = m.predict(df)
    y_test, p_test = test["sales"].to_numpy(), pred_all[-test_weeks:]
    return {
        "train_r2": m.fit_metrics["r2"], "train_mape": m.fit_metrics["mape"],
        "test_r2": r2(y_test, p_test), "test_mape": mape(y_test, p_test),
        "test_weeks": test_weeks,
        "series": pd.DataFrame({"date": df["date"], "actual": df["sales"], "predicted": pred_all,
                                "split": ["train"] * len(train) + ["test"] * test_weeks}),
    }


def business_value(model: MMM, total_budget: float, n_weeks: int = 13, df=None, **opt_kwargs) -> dict:
    """2) Business value: current split vs optimized split, same budget."""
    cur = evaluate_plan(model, current_allocation(model, total_budget, df=df), n_weeks, "current split")
    best = optimize(model, total_budget, n_weeks, **opt_kwargs)
    return {
        "budget": total_budget, "n_weeks": n_weeks,
        "current_sales": cur.total_sales, "optimized_sales": best.total_sales,
        "uplift": best.total_sales - cur.total_sales,
        "uplift_pct": best.total_sales / cur.total_sales - 1 if cur.total_sales else 0.0,
        "current": cur.to_dict(), "optimized": best.to_dict(),
    }


def ground_truth_recovery(sim: SimulationResult | None = None, **mmm_kwargs) -> dict:
    """3) Validation: on simulated data we know each channel's true ROI; does the model find it?"""
    sim = sim or simulate()
    m = MMM(**mmm_kwargs).fit(sim.data)
    est = m.channel_summary(sim.data).set_index("channel")["roi"].to_dict()
    tru = true_roi(sim)
    rows = [{"channel": c, "true_roi": tru[c], "estimated_roi": est[c],
             "abs_pct_error": abs(est[c] - tru[c]) / tru[c]} for c in tru]
    t = np.array([r["true_roi"] for r in rows])
    e = np.array([r["estimated_roi"] for r in rows])
    rank_corr = float(pd.Series(t).rank().corr(pd.Series(e).rank()))

    # Decision quality: how much of the truly achievable uplift does our plan capture?
    budget = float(sim.data[[c for c in sim.data if c.startswith("spend_")]].tail(13).sum().sum())
    true_model = _truth_as_model(sim, m)
    cur = evaluate_plan(true_model, current_allocation(m, budget), 13)
    ours = evaluate_plan(true_model, optimize(m, budget).allocation, 13)
    oracle = optimize(true_model, budget)
    achievable = oracle.total_sales - cur.total_sales
    return {
        "channels": rows,
        "mean_abs_pct_error": float(np.mean([r["abs_pct_error"] for r in rows])),
        "roi_rank_correlation": rank_corr,
        "fit_r2": m.fit_metrics["r2"],
        "true_uplift_of_our_plan": ours.total_sales - cur.total_sales,
        "best_possible_uplift": achievable,
        "share_of_best_uplift_captured": (ours.total_sales - cur.total_sales) / achievable if achievable > 0 else None,
    }


def _truth_as_model(sim: SimulationResult, fitted: MMM) -> MMM:
    from .mmm import ChannelParams
    tm = MMM()
    tm.channels = list(fitted.channels)
    tm.spend_stats = fitted.spend_stats
    tm.params = {c: ChannelParams(p.decay, p.half_sat, p.slope, p.beta) for c, p in sim.truth.items()}
    return tm
