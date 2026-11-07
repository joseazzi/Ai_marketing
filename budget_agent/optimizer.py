"""Budget optimizer (SciPy) and what-if scenarios on top of a fitted MMM.

The plan is expressed as total spend per channel over ``n_weeks``; spend is
assumed to be spread evenly over the weeks, so each channel's expected
incremental sales are ``n_weeks * response(budget / n_weeks)`` (steady state).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import minimize

from .mmm import MMM


@dataclass
class Plan:
    name: str
    n_weeks: int
    allocation: dict[str, float]          # total spend per channel for the period
    expected_sales: dict[str, float]      # incremental sales per channel for the period
    marginal_roi: dict[str, float]        # at the planned weekly spend
    saturation_pct: dict[str, float]

    @property
    def total_budget(self) -> float:
        return float(sum(self.allocation.values()))

    @property
    def total_sales(self) -> float:
        return float(sum(self.expected_sales.values()))

    @property
    def roi(self) -> float:
        return self.total_sales / self.total_budget if self.total_budget else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(total_budget=round(self.total_budget, 2), total_incremental_sales=round(self.total_sales, 2),
                 roi=round(self.roi, 3))
        for k in ("allocation", "expected_sales"):
            d[k] = {c: round(v, 2) for c, v in d[k].items()}
        for k in ("marginal_roi", "saturation_pct"):
            d[k] = {c: round(v, 3) for c, v in d[k].items()}
        return d


def evaluate_plan(model: MMM, allocation: dict[str, float], n_weeks: int, name: str = "plan") -> Plan:
    alloc = {ch: float(max(allocation.get(ch, 0.0), 0.0)) for ch in model.channels}
    sales, mroi, sat = {}, {}, {}
    for ch, total in alloc.items():
        p = model.params[ch]
        w = total / n_weeks
        sales[ch] = float(n_weeks * p.response(w))
        mroi[ch] = float(p.marginal_roi(w))
        sat[ch] = float(p.response(w) / p.beta) if p.beta > 0 else 0.0
    return Plan(name, n_weeks, alloc, sales, mroi, sat)


def current_allocation(model: MMM, total_budget: float, recent_weeks: int = 13, df=None) -> dict[str, float]:
    """Today's split: the share of spend per channel in recent history, scaled to the budget."""
    if df is not None:
        from .data import SPEND_PREFIX
        shares = {ch: float(df[SPEND_PREFIX + ch].tail(recent_weeks).sum()) for ch in model.channels}
    else:
        shares = {ch: model.spend_stats[ch]["last13_mean"] for ch in model.channels}
    tot = sum(shares.values()) or 1.0
    return {ch: total_budget * v / tot for ch, v in shares.items()}


def optimize(
    model: MMM,
    total_budget: float,
    n_weeks: int = 13,
    min_spend: dict[str, float] | None = None,
    max_spend: dict[str, float] | None = None,
    max_change_vs: dict[str, float] | None = None,
    max_change_pct: float | None = None,
    goal: str = "max_sales",
    target_roi: float | None = None,
    n_starts: int = 8,
    seed: int = 0,
) -> Plan:
    """Find the split of ``total_budget`` that maximises incremental sales.

    * ``min_spend`` / ``max_spend``: per-channel limits on total period spend.
    * ``max_change_vs`` + ``max_change_pct``: keep each channel within +/- pct of a
      reference allocation (useful for "don't change anything by more than 30%").
    * ``goal="max_sales"`` spends the whole budget; ``goal="min_budget_for_roi"``
      allows spending less when extra dollars fall below ``target_roi`` marginal return.
    """
    chans = model.channels
    n = len(chans)
    lo = np.array([float((min_spend or {}).get(c, 0.0)) for c in chans])
    hi = np.array([float((max_spend or {}).get(c, total_budget)) for c in chans])
    if max_change_vs and max_change_pct is not None:
        ref = np.array([max_change_vs.get(c, 0.0) for c in chans])
        lo = np.maximum(lo, ref * (1 - max_change_pct))
        hi = np.minimum(hi, ref * (1 + max_change_pct))
    hi = np.minimum(hi, total_budget)
    if lo.sum() > total_budget + 1e-6:
        raise ValueError(f"Minimum spends ({lo.sum():,.0f}) exceed the budget ({total_budget:,.0f}).")
    if goal == "max_sales" and hi.sum() < total_budget - 1e-6:
        raise ValueError(f"Maximum spends ({hi.sum():,.0f}) cannot absorb the budget ({total_budget:,.0f}).")

    scale = total_budget
    params = [model.params[c] for c in chans]
    penalty_roi = target_roi if goal == "min_budget_for_roi" else 0.0

    def neg_obj(z):
        x = z * scale / n_weeks
        val = sum(p.response(xi) for p, xi in zip(params, x)) * n_weeks
        return -(val - penalty_roi * z.sum() * scale) / scale

    def neg_grad(z):
        x = z * scale / n_weeks
        g = np.array([p.marginal_roi(xi) for p, xi in zip(params, x)]) - penalty_roi
        return -g

    bounds = list(zip(lo / scale, hi / scale))
    if goal == "max_sales":
        cons = [{"type": "eq", "fun": lambda z: z.sum() - 1.0, "jac": lambda z: np.ones(n)}]
    else:
        cons = [{"type": "ineq", "fun": lambda z: 1.0 - z.sum(), "jac": lambda z: -np.ones(n)}]

    rng = np.random.default_rng(seed)
    starts = [np.clip(np.full(n, 1.0 / n), lo / scale, hi / scale)]
    ref_cur = current_allocation(model, total_budget)
    starts.append(np.clip(np.array([ref_cur[c] for c in chans]) / scale, lo / scale, hi / scale))
    for _ in range(n_starts):
        starts.append(np.clip(rng.dirichlet(np.ones(n)), lo / scale, hi / scale))

    best = None
    for z0 in starts:
        r = minimize(neg_obj, z0, jac=neg_grad, bounds=bounds, constraints=cons, method="SLSQP",
                     options={"maxiter": 500, "ftol": 1e-12})
        feasible = r.success or abs(r.x.sum() - 1) < 1e-4
        if feasible and (best is None or r.fun < best.fun):
            best = r
    if best is None:
        raise RuntimeError("Optimizer failed to find a feasible allocation.")
    alloc = dict(zip(chans, np.clip(best.x, lo / scale, hi / scale) * scale))
    return evaluate_plan(model, alloc, n_weeks, name="optimized")


def what_if(
    model: MMM,
    base: dict[str, float],
    n_weeks: int,
    changes: dict[str, float],
    reallocate: bool = False,
) -> Plan:
    """Apply relative changes (e.g. ``{"tv": -0.30}``) to a base allocation.

    With ``reallocate=True`` money freed (or needed) is spread across the other
    channels in proportion to their marginal ROI, keeping the total constant.
    """
    new = {c: float(v) for c, v in base.items()}
    delta = 0.0
    for ch, pct in changes.items():
        if ch not in new:
            raise ValueError(f"Unknown channel '{ch}'. Known: {list(new)}")
        d = new[ch] * pct
        new[ch] += d
        delta -= d
    if reallocate and abs(delta) > 1e-9:
        others = [c for c in new if c not in changes]
        weights = np.array([max(model.params[c].marginal_roi(new[c] / n_weeks), 1e-9) for c in others])
        weights = weights / weights.sum()
        for c, w in zip(others, weights):
            new[c] = max(new[c] + delta * w, 0.0)
    label = ", ".join(f"{c} {p:+.0%}" for c, p in changes.items()) + (" (reallocated)" if reallocate else "")
    return evaluate_plan(model, new, n_weeks, name=f"what-if: {label}")


def compare(plans: list[Plan]) -> list[dict]:
    base = plans[0]
    out = []
    for p in plans:
        out.append({
            "name": p.name,
            "total_budget": round(p.total_budget, 2),
            "incremental_sales": round(p.total_sales, 2),
            "roi": round(p.roi, 3),
            "sales_vs_first": round(p.total_sales - base.total_sales, 2),
            "sales_vs_first_pct": round((p.total_sales / base.total_sales - 1) if base.total_sales else 0.0, 4),
            "allocation": {c: round(v, 2) for c, v in p.allocation.items()},
        })
    return out


def explain_shifts(model: MMM, current: Plan, optimized: Plan, top: int = 3) -> list[dict]:
    """Model-grounded reasons for the biggest budget moves (used by agent + autopilot)."""
    moves = []
    for ch in model.channels:
        d = optimized.allocation[ch] - current.allocation[ch]
        moves.append({
            "channel": ch,
            "change": round(d, 2),
            "change_pct": round(d / current.allocation[ch], 3) if current.allocation[ch] else None,
            "current_marginal_roi": round(current.marginal_roi[ch], 3),
            "optimized_marginal_roi": round(optimized.marginal_roi[ch], 3),
            "current_saturation_pct": round(current.saturation_pct[ch], 3),
            "reason": (
                "saturated: the next dollar returns little, so budget moves out"
                if d < 0 and current.saturation_pct[ch] > 0.6 else
                "low marginal return compared with other channels" if d < 0 else
                "room to grow: high marginal return and not yet saturated"
            ),
        })
    moves.sort(key=lambda m: abs(m["change"]), reverse=True)
    return moves[:top]
