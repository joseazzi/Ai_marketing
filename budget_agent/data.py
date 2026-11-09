"""Data layer: loading, cleaning, quality checks and a simulator with known ground truth.

Expected CSV format (one row per week):

    date,sales,spend_facebook,spend_instagram,spend_google_ads,spend_tv,...,holiday,promo

* ``date``      - week start date (any format pandas can parse)
* ``sales``     - weekly sales in currency units
* ``spend_<x>`` - weekly spend on channel ``x`` (one column per channel)
* anything else numeric (e.g. ``holiday``, ``promo``, ``price_index``) is treated
  as an external control variable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SPEND_PREFIX = "spend_"
DATE_COL = "date"
TARGET_COL = "sales"
RECOMMENDED_MIN_WEEKS = 104  # MMM needs ~2 years of weekly data


# --------------------------------------------------------------------------- #
# Loading & cleaning
# --------------------------------------------------------------------------- #
def channels_of(df: pd.DataFrame) -> list[str]:
    return [c[len(SPEND_PREFIX):] for c in df.columns if c.startswith(SPEND_PREFIX)]


def controls_of(df: pd.DataFrame) -> list[str]:
    skip = {DATE_COL, TARGET_COL}
    return [
        c for c in df.columns
        if c not in skip and not c.startswith(SPEND_PREFIX) and pd.api.types.is_numeric_dtype(df[c])
    ]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, parse dates, sort, de-duplicate and fill gaps."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if DATE_COL not in df.columns:
        for alt in ("week", "week_start", "ds", "period"):
            if alt in df.columns:
                df = df.rename(columns={alt: DATE_COL})
                break
    missing = {DATE_COL, TARGET_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {sorted(missing)}")
    if not channels_of(df):
        raise ValueError(f"No spend columns found. Name them '{SPEND_PREFIX}<channel>'.")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).drop_duplicates(subset=DATE_COL, keep="last")
    for c in df.columns:
        if c != DATE_COL:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    spend_cols = [SPEND_PREFIX + ch for ch in channels_of(df)]
    df[spend_cols] = df[spend_cols].fillna(0.0).clip(lower=0.0)
    ctrl = controls_of(df)
    df[ctrl] = df[ctrl].fillna(0.0)
    df = df.dropna(subset=[TARGET_COL])
    return df.reset_index(drop=True)


def load_csv(path_or_buffer) -> pd.DataFrame:
    return clean(pd.read_csv(path_or_buffer))


def quality_report(df: pd.DataFrame) -> dict:
    """Checks the agent runs *before* trusting any model built on this data."""
    issues: list[str] = []
    warnings: list[str] = []
    n = len(df)
    chans = channels_of(df)

    if n < 52:
        issues.append(f"Only {n} weeks of data; at least 52 are required, {RECOMMENDED_MIN_WEEKS} recommended.")
    elif n < RECOMMENDED_MIN_WEEKS:
        warnings.append(f"{n} weeks of data; {RECOMMENDED_MIN_WEEKS}+ recommended for reliable MMM.")

    gaps = df[DATE_COL].diff().dt.days.dropna()
    irregular = int((gaps != 7).sum())
    if irregular:
        warnings.append(f"{irregular} irregular gaps between weeks (expected 7 days).")

    zero_var = [ch for ch in chans if df[SPEND_PREFIX + ch].std() == 0]
    if zero_var:
        issues.append(f"Channels with constant spend cannot be measured: {zero_var}")

    rarely = [ch for ch in chans if (df[SPEND_PREFIX + ch] > 0).mean() < 0.2]
    if rarely:
        warnings.append(f"Channels active in fewer than 20% of weeks (estimates will be uncertain): {rarely}")

    corr = df[[SPEND_PREFIX + c for c in chans]].corr().abs()
    collinear = [
        (chans[i], chans[j], round(float(corr.iloc[i, j]), 2))
        for i in range(len(chans)) for j in range(i + 1, len(chans))
        if corr.iloc[i, j] > 0.85
    ]
    if collinear:
        warnings.append(f"Highly correlated channel pairs (hard to separate): {collinear}")
    moderate = [
        (chans[i], chans[j], round(float(corr.iloc[i, j]), 2))
        for i in range(len(chans)) for j in range(i + 1, len(chans))
        if corr.iloc[i, j] <= 0.85 and round(float(corr.iloc[i, j]), 2) >= 0.8  # compared as displayed (0.80)
    ]
    if moderate:
        warnings.append(f"Moderately correlated channel pairs (harder to separate): {moderate}")

    z = (df[TARGET_COL] - df[TARGET_COL].mean()) / (df[TARGET_COL].std() or 1)
    outliers = int((z.abs() > 4).sum())
    if outliers:
        warnings.append(f"{outliers} extreme sales outliers (|z|>4).")

    return {
        "weeks": n,
        "start": str(df[DATE_COL].min().date()),
        "end": str(df[DATE_COL].max().date()),
        "channels": chans,
        "controls": controls_of(df),
        "total_spend": round(float(df[[SPEND_PREFIX + c for c in chans]].sum().sum()), 2),
        "total_sales": round(float(df[TARGET_COL].sum()), 2),
        "issues": issues,
        "warnings": warnings,
        "ok_to_model": not issues,
    }


# --------------------------------------------------------------------------- #
# Simulator with known ground truth (used for validation and as backup data)
# --------------------------------------------------------------------------- #
@dataclass
class ChannelTruth:
    decay: float        # adstock carry-over per week
    half_sat: float     # weekly spend at which half the max effect is reached
    slope: float        # Hill slope
    beta: float         # max weekly sales the channel can drive
    mean_spend: float   # typical weekly spend
    pulse: float = 0.0  # 0 = always-on, >0 = probability of an "off" week (flighting)


DEFAULT_TRUTH: dict[str, ChannelTruth] = {
    "facebook":   ChannelTruth(decay=0.30, half_sat=1800, slope=1.2, beta=6500,  mean_spend=1500),
    "instagram":  ChannelTruth(decay=0.25, half_sat=2600, slope=1.1, beta=9000,  mean_spend=1000),
    "google_ads": ChannelTruth(decay=0.10, half_sat=1500, slope=1.4, beta=7000,  mean_spend=1800),
    "tv":         ChannelTruth(decay=0.65, half_sat=1200, slope=2.0, beta=6000,  mean_spend=2600, pulse=0.35),
    "radio":      ChannelTruth(decay=0.45, half_sat=900,  slope=1.6, beta=1600,  mean_spend=700,  pulse=0.25),
    "email":      ChannelTruth(decay=0.05, half_sat=150,  slope=1.0, beta=1400,  mean_spend=200),
}


@dataclass
class SimulationResult:
    data: pd.DataFrame
    truth: dict[str, ChannelTruth]
    true_contributions: pd.DataFrame
    shifted_truth: dict[str, ChannelTruth] | None = None
    shift_week: int | None = None
    meta: dict = field(default_factory=dict)


def _adstock(x: np.ndarray, decay: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for t, v in enumerate(x):
        carry = v + decay * carry
        out[t] = carry * (1 - decay)
    return out


def _hill(x: np.ndarray, k: float, s: float) -> np.ndarray:
    x = np.maximum(x, 0)
    return x**s / (x**s + k**s)


def simulate(
    weeks: int = 156,
    seed: int = 7,
    start: str = "2023-01-02",
    truth: dict[str, ChannelTruth] | None = None,
    shift_week: int | None = None,
    shift: dict[str, dict] | None = None,
    noise: float = 0.025,
) -> SimulationResult:
    """Generate weekly spend + sales with *known* channel effects.

    ``shift_week``/``shift`` let you change the true effects from a given week on
    (e.g. ``{"instagram": {"beta": 4000}}``) to demo the agent learning from new data.
    """
    rng = np.random.default_rng(seed)
    truth = {k: ChannelTruth(**vars(v)) for k, v in (truth or DEFAULT_TRUTH).items()}
    shifted = None
    if shift_week is not None and shift:
        shifted = {k: ChannelTruth(**{**vars(v), **shift.get(k, {})}) for k, v in truth.items()}

    dates = pd.date_range(start, periods=weeks, freq="W-MON")
    t = np.arange(weeks)
    woy = dates.isocalendar().week.to_numpy().astype(float)

    # External factors
    holiday = np.isin(woy, [1, 14, 22, 47, 48, 51, 52]).astype(float)
    promo = (rng.random(weeks) < 0.12).astype(float)
    season = 0.10 * np.sin(2 * np.pi * woy / 52) + 0.05 * np.cos(4 * np.pi * woy / 52)

    # Spend: slow budget drift + noise + flighting for TV/radio
    df = pd.DataFrame({DATE_COL: dates})
    for ch, p in truth.items():
        drift = 1 + 0.25 * np.sin(2 * np.pi * (t / 78.0) + rng.uniform(0, 2 * np.pi))
        s = p.mean_spend * drift * rng.lognormal(0, 0.5, weeks)
        if p.pulse:
            s = s * (rng.random(weeks) > p.pulse) * (1 / (1 - p.pulse))
        df[SPEND_PREFIX + ch] = np.round(s, 2)

    baseline = 30000 * (1 + 0.0015 * t) * (1 + season) + 6000 * holiday + 3500 * promo
    contrib = pd.DataFrame({DATE_COL: dates})
    media = np.zeros(weeks)
    for ch in truth:
        x = df[SPEND_PREFIX + ch].to_numpy()
        ad = None
        segs = []
        for regime in ([truth] if shifted is None else [truth, shifted]):
            p = regime[ch]
            ad = _adstock(x, p.decay)
            segs.append(p.beta * _hill(ad, p.half_sat, p.slope))
        c = segs[0] if shifted is None else np.where(t < shift_week, segs[0], segs[1])
        contrib[ch] = c
        media += c

    sales = (baseline + media) * rng.lognormal(0, noise, weeks)
    df[TARGET_COL] = np.round(sales, 2)
    df["holiday"] = holiday
    df["promo"] = promo
    df = df[[DATE_COL, TARGET_COL] + [SPEND_PREFIX + c for c in truth] + ["holiday", "promo"]]
    contrib["baseline"] = baseline
    return SimulationResult(df, truth, contrib, shifted, shift_week, {"seed": seed, "weeks": weeks})


def true_roi(sim: SimulationResult, start: int = 0, end: int | None = None) -> dict[str, float]:
    d = sim.data.iloc[start:end]
    c = sim.true_contributions.iloc[start:end]
    return {
        ch: float(c[ch].sum() / max(d[SPEND_PREFIX + ch].sum(), 1e-9))
        for ch in sim.truth
    }
