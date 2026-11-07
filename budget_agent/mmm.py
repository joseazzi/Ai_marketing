"""Media Mix Model (MMM) with geometric adstock and Hill saturation.

    sales_t = intercept + trend*t + seasonality_t + controls_t*gamma
              + sum_c beta_c * Hill(Adstock(spend_c)_t; K_c, S_c) + noise

* Adstock (carry-over): a_t = (1-d) * (x_t + d*x_{t-1} + d^2*x_{t-2} + ...)
  normalised so that a constant weekly spend x has steady-state adstock x.
* Hill saturation: h(x) = x^S / (x^S + K^S); beta_c is the channel's maximum
  weekly sales effect, K_c the weekly spend that reaches half of it.

Fitting: the non-linear parameters (decay, K, S per channel) are searched with
SciPy's differential evolution; for each candidate the linear parameters are
solved exactly by bounded least squares (media betas >= 0). Optional recency
weighting lets the model adapt when the market changes.

This is a transparent, dependency-light implementation of the same model family
used by Google Meridian and Meta Robyn; see README for how to swap them in.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, differential_evolution, lsq_linear
from scipy.signal import lfilter

from .data import DATE_COL, SPEND_PREFIX, TARGET_COL, channels_of, controls_of


# --------------------------------------------------------------------------- #
# Transformations
# --------------------------------------------------------------------------- #
def adstock(x: np.ndarray, decay: float) -> np.ndarray:
    return lfilter([1 - decay], [1, -decay], np.asarray(x, dtype=float))


def hill(x, k: float, s: float):
    x = np.maximum(np.asarray(x, dtype=float), 0.0)
    xs = x**s
    return xs / (xs + k**s)


def hill_derivative(x, k: float, s: float):
    x = np.maximum(np.asarray(x, dtype=float), 1e-9)
    return s * x ** (s - 1) * k**s / (x**s + k**s) ** 2


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class ChannelParams:
    decay: float
    half_sat: float
    slope: float
    beta: float

    def response(self, weekly_spend):
        """Steady-state weekly incremental sales for a constant weekly spend."""
        return self.beta * hill(weekly_spend, self.half_sat, self.slope)

    def marginal_roi(self, weekly_spend):
        """Extra sales from the next $1 of weekly spend."""
        return self.beta * hill_derivative(weekly_spend, self.half_sat, self.slope)


@dataclass
class MMM:
    recency_halflife: float | None = None  # weeks; None = all weeks weigh the same
    n_fourier: int = 2
    seed: int = 0
    maxiter: int = 60
    min_baseline_share: float = 0.3  # prior: at least this share of average sales is organic (not ads)

    # learned
    channels: list[str] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)
    params: dict[str, ChannelParams] = field(default_factory=dict)
    linear_coefs: dict[str, float] = field(default_factory=dict)
    t0: str | None = None
    spend_stats: dict[str, dict] = field(default_factory=dict)
    fit_metrics: dict = field(default_factory=dict)
    trained_weeks: int = 0

    # ------------------------------------------------------------------ #
    def _base_design(self, df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
        dates = pd.to_datetime(df[DATE_COL])
        t = ((dates - pd.Timestamp(self.t0)).dt.days / 7.0).to_numpy()
        doy = dates.dt.dayofyear.to_numpy()
        cols, names = [np.ones(len(df)), t / 52.0], ["intercept", "trend"]
        for k in range(1, self.n_fourier + 1):
            cols += [np.sin(2 * np.pi * k * doy / 365.25), np.cos(2 * np.pi * k * doy / 365.25)]
            names += [f"sin{k}", f"cos{k}"]
        for c in self.controls:
            cols.append(df[c].to_numpy(dtype=float))
            names.append(c)
        return np.column_stack(cols), names

    def _media_features(self, df: pd.DataFrame, theta: np.ndarray) -> np.ndarray:
        feats = []
        for i, ch in enumerate(self.channels):
            d, k, s = theta[3 * i: 3 * i + 3]
            feats.append(hill(adstock(df[SPEND_PREFIX + ch].to_numpy(), d), k, s))
        return np.column_stack(feats)

    def _bounds(self) -> list[tuple[float, float]]:
        b = []
        for ch in self.channels:
            m = self.spend_stats[ch]["mean_active"]
            b += [(0.0, 0.85), (0.3 * m, 4.0 * m), (0.7, 3.0)]
        return b

    # ------------------------------------------------------------------ #
    def fit(self, df: pd.DataFrame, prior: "MMM | None" = None, prior_strength: float = 1.0,
            verbose: bool = False) -> "MMM":
        """Fit the model. With ``prior`` (the previous model) this is an *update*: the search
        starts from the previous solution and is pulled towards it, so estimates only move as
        far as the new evidence justifies (a light-weight Bayesian-style update)."""
        df = df.reset_index(drop=True)
        if prior is not None and set(prior.channels) != set(channels_of(df)):
            prior = None
        self.channels = channels_of(df)
        self.controls = controls_of(df)
        self.t0 = str(pd.to_datetime(df[DATE_COL]).min().date())
        self.trained_weeks = len(df)
        self.spend_stats = {}
        for ch in self.channels:
            x = df[SPEND_PREFIX + ch]
            self.spend_stats[ch] = {
                "mean": float(x.mean()),
                "mean_active": float(x[x > 0].mean()) if (x > 0).any() else 1.0,
                "max": float(x.max()),
                "last13_mean": float(x.tail(13).mean()),
            }

        y = df[TARGET_COL].to_numpy(dtype=float)
        y_scale = float(np.mean(y))
        yn = y / y_scale
        Xb, base_names = self._base_design(df)
        n_base = Xb.shape[1]

        if self.recency_halflife:
            age = np.arange(len(df))[::-1]
            w = 0.5 ** (age / self.recency_halflife)
        else:
            w = np.ones(len(df))
        sw = np.sqrt(w / w.mean())

        lb = np.r_[np.full(n_base, -np.inf), np.zeros(len(self.channels))]
        ub = np.full(n_base + len(self.channels), np.inf)
        lb[0] = self.min_baseline_share  # intercept floor (sales are normalised to mean 1)

        n_lin = n_base + len(self.channels)
        if prior is not None:
            # pseudo-observations pulling each media beta towards the previous estimate (relative scale)
            lam_b = 0.05 * prior_strength
            b0 = np.array([max(prior.params[c].beta, 1e-6) / y_scale for c in self.channels])
            A_prior = np.zeros((len(self.channels), n_lin))
            A_prior[np.arange(len(self.channels)), n_base + np.arange(len(self.channels))] = np.sqrt(lam_b) / b0
            y_prior = np.full(len(self.channels), np.sqrt(lam_b))

        def solve(theta):
            X = np.column_stack([Xb, self._media_features(df, theta)])
            A, b = X * sw[:, None], yn * sw
            if prior is not None:
                A, b = np.vstack([A, A_prior]), np.r_[b, y_prior]
            res = lsq_linear(A, b, bounds=(lb, ub), method="bvls")
            return res, X

        def loss(theta):
            res, _ = solve(theta)
            # weak prior: keep K near typical spend and slope near 1.5 (helps identifiability)
            pen = 0.0
            for i, ch in enumerate(self.channels):
                m = self.spend_stats[ch]["mean_active"]
                pen += 2e-5 * np.log(theta[3 * i + 1] / m) ** 2 + 1e-5 * (theta[3 * i + 2] - 1.5) ** 2
                if prior is not None:
                    p0 = prior.params[ch]
                    pen += prior_nl * (np.log(theta[3 * i + 1] / p0.half_sat) ** 2
                                       + (theta[3 * i] - p0.decay) ** 2 + 0.25 * (theta[3 * i + 2] - p0.slope) ** 2)
            return float(np.sum(res.fun**2)) / len(yn) + pen

        bounds = self._bounds()
        x0 = None
        prior_nl = 0.0
        if prior is not None:
            prior_nl = 5e-4 * prior_strength / len(self.channels)
            x0 = np.clip(np.concatenate([[prior.params[c].decay, prior.params[c].half_sat, prior.params[c].slope]
                                         for c in self.channels]), [b[0] for b in bounds], [b[1] for b in bounds])
        result = differential_evolution(
            loss, bounds, seed=self.seed, maxiter=self.maxiter, popsize=12, x0=x0,
            tol=1e-7, polish=True, disp=verbose, updating="deferred",
        )
        res, X = solve(result.x)
        coefs = res.x * y_scale
        self.linear_coefs = dict(zip(base_names, map(float, coefs[:n_base])))
        self.params = {}
        for i, ch in enumerate(self.channels):
            d, k, s = result.x[3 * i: 3 * i + 3]
            self.params[ch] = ChannelParams(float(d), float(k), float(s), float(coefs[n_base + i]))

        pred = self.predict(df)
        self.fit_metrics = {"r2": r2(y, pred), "mape": mape(y, pred), "weeks": len(df)}
        return self

    # ------------------------------------------------------------------ #
    def decompose(self, df: pd.DataFrame) -> pd.DataFrame:
        """Weekly sales split into baseline + each channel's contribution."""
        Xb, names = self._base_design(df)
        base = Xb @ np.array([self.linear_coefs[n] for n in names])
        out = pd.DataFrame({DATE_COL: pd.to_datetime(df[DATE_COL]).to_numpy(), "baseline": base})
        for ch, p in self.params.items():
            out[ch] = p.beta * hill(adstock(df[SPEND_PREFIX + ch].to_numpy(), p.decay), p.half_sat, p.slope)
        return out

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        d = self.decompose(df)
        return d.drop(columns=[DATE_COL]).sum(axis=1).to_numpy()

    def baseline_forecast(self, start: str | pd.Timestamp, n_weeks: int) -> float:
        """Expected non-media sales for an upcoming period (no holidays/promos assumed)."""
        dates = pd.date_range(pd.Timestamp(start), periods=n_weeks, freq="7D")
        fut = pd.DataFrame({DATE_COL: dates, **{c: 0.0 for c in self.controls}})
        Xb, names = self._base_design(fut)
        return float((Xb @ np.array([self.linear_coefs[n] for n in names])).sum())

    # ------------------------------------------------------------------ #
    def saturation_point(self, ch: str, breakeven: float = 1.0) -> float | None:
        """Weekly spend above which the next $1 returns less than ``breakeven`` in sales."""
        p = self.params[ch]
        f = lambda x: p.marginal_roi(x) - breakeven  # noqa: E731
        # marginal ROI peaks at the inflection point (s>1) or at 0 (s<=1)
        x_peak = p.half_sat * ((p.slope - 1) / (p.slope + 1)) ** (1 / p.slope) if p.slope > 1 else 1e-6
        hi = p.half_sat * 50
        if f(max(x_peak, 1e-6)) <= 0:
            return 0.0  # never profitable at this break-even level
        if f(hi) > 0:
            return None
        return float(brentq(f, max(x_peak, 1e-6), hi))

    def channel_summary(self, df: pd.DataFrame, recent_weeks: int = 13, breakeven: float = 1.0) -> pd.DataFrame:
        dec = self.decompose(df)
        rows = []
        for ch, p in self.params.items():
            spend = df[SPEND_PREFIX + ch]
            contrib = dec[ch].sum()
            cur = float(spend.tail(recent_weeks).mean())
            sat = self.saturation_point(ch, breakeven)
            rows.append({
                "channel": ch,
                "total_spend": float(spend.sum()),
                "contribution": float(contrib),
                "share_of_spend": float(spend.sum() / max(df[[SPEND_PREFIX + c for c in self.channels]].sum().sum(), 1e-9)),
                "share_of_media_sales": 0.0,
                "roi": float(contrib / max(spend.sum(), 1e-9)),
                "current_weekly_spend": cur,
                "marginal_roi": float(p.marginal_roi(cur)),
                "saturation_pct": float(hill(cur, p.half_sat, p.slope)),
                "saturation_weekly_spend": sat,
                "carryover_weeks": float(1 / (1 - p.decay)) if p.decay < 1 else None,
                "decay": p.decay, "half_sat": p.half_sat, "slope": p.slope, "beta": p.beta,
            })
        out = pd.DataFrame(rows)
        out["share_of_media_sales"] = out["contribution"] / out["contribution"].sum()
        return out.sort_values("roi", ascending=False).reset_index(drop=True)

    def response_curves(self, max_multiple: float = 3.0, points: int = 60) -> pd.DataFrame:
        rows = []
        for ch, p in self.params.items():
            top = max(self.spend_stats[ch]["max"], p.half_sat) * max_multiple / 2
            for x in np.linspace(0, top, points):
                rows.append({"channel": ch, "weekly_spend": x, "weekly_sales": float(p.response(x)),
                             "marginal_roi": float(p.marginal_roi(x))})
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        d = asdict(self)
        d["params"] = {k: asdict(v) for k, v in self.params.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MMM":
        d = dict(d)
        params = {k: ChannelParams(**v) for k, v in d.pop("params").items()}
        m = cls(**d)
        m.params = params
        return m

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "MMM":
        return cls.from_dict(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def r2(y, yhat) -> float:
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def mape(y, yhat) -> float:
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(np.mean(np.abs((y - yhat) / y)))
