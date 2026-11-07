"""Convert the public Conjura e-commerce MMM dataset into this project's weekly CSV format.

Source: "Multi-Region Marketing Mix Modeling (MMM) Dataset for Several eCommerce Brands" (Conjura, Figshare
25314841): real, anonymised daily data per brand: purchases, discounts and ad spend on Google/Meta/TikTok.
https://figshare.com/articles/dataset/Multi-Region_Marketing_Mix_Modeling_MMM_Dataset_for_Several_eCommerce_Brands/25314841

Usage:
    python scripts/prepare_conjura.py conjura_mmm_data.csv --list
    python scripts/prepare_conjura.py conjura_mmm_data.csv --brand 826fd57057c52458a1f77cae4e3e6c67 -o data/real/brand.csv

Output columns:
    date, sales (net revenue = original price - discounts), spend_<channel> ..., holiday, discount_rate
"""
from __future__ import annotations

import argparse

import pandas as pd

MIN_SHARE = 0.03      # channel must carry >= 3% of total spend ...
MIN_ACTIVE = 0.5      # ... and be active in >= 50% of weeks, otherwise it is folded into "other_paid"
HOLIDAY_WEEKS = {47, 48, 51, 52}  # Black Friday / Cyber Monday / Christmas


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d["TERRITORY_NAME"] == "All Territories"].copy()
    d["DATE_DAY"] = pd.to_datetime(d["DATE_DAY"])
    return d


def spend_cols(d: pd.DataFrame) -> list[str]:
    return [c for c in d.columns if c.endswith("_SPEND")]


def weekly(d: pd.DataFrame, brand: str) -> pd.DataFrame:
    g = d[d["MMM_TIMESERIES_ID"] == brand].set_index("DATE_DAY").sort_index()
    if g.empty:
        raise SystemExit(f"Brand {brand} not found")
    sp = spend_cols(d)
    cols = sp + ["ALL_PURCHASES_ORIGINAL_PRICE", "ALL_PURCHASES_GROSS_DISCOUNT"]
    w = g[cols].fillna(0).resample("W-MON", label="left", closed="left").sum()
    w = w.iloc[1:-1]  # drop partial first/last weeks

    total = w[sp].sum().sum()
    keep = [c for c in sp if (w[c] > 0).mean() >= MIN_ACTIVE and w[c].sum() >= MIN_SHARE * total]
    rest = [c for c in sp if c not in keep]

    out = pd.DataFrame(index=w.index)
    out["sales"] = (w["ALL_PURCHASES_ORIGINAL_PRICE"] - w["ALL_PURCHASES_GROSS_DISCOUNT"]).round(2)
    for c in keep:
        out["spend_" + c.removesuffix("_SPEND").lower()] = w[c].round(2)
    if rest and w[rest].sum().sum() > 0.01 * total:
        out["spend_other_paid"] = w[rest].sum(axis=1).round(2)
    out["holiday"] = out.index.isocalendar().week.isin(HOLIDAY_WEEKS).astype(float).to_numpy()
    price = w["ALL_PURCHASES_ORIGINAL_PRICE"].replace(0, pd.NA)
    out["discount_rate"] = (w["ALL_PURCHASES_GROSS_DISCOUNT"] / price).fillna(0).astype(float).round(4)
    out = out[out["sales"] > 0]
    return out.reset_index(names="date")


def list_brands(d: pd.DataFrame, min_weeks: int = 104) -> pd.DataFrame:
    rows = []
    for tid, g in d.groupby("MMM_TIMESERIES_ID"):
        w = weekly(d, tid)
        ch = [c for c in w.columns if c.startswith("spend_")]
        rows.append({"brand": tid, "vertical": g["ORGANISATION_VERTICAL"].iloc[0],
                     "subvertical": g["ORGANISATION_SUBVERTICAL"].iloc[0], "currency": g["CURRENCY_CODE"].iloc[0],
                     "weeks": len(w), "start": w["date"].min().date(), "end": w["date"].max().date(),
                     "channels": len(ch), "channel_names": ",".join(c[6:] for c in ch),
                     "total_spend": round(w[ch].sum().sum()), "net_revenue": round(w["sales"].sum())})
    r = pd.DataFrame(rows)
    return r[r["weeks"] >= min_weeks].sort_values(["channels", "weeks"], ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--brand")
    ap.add_argument("-o", "--out")
    a = ap.parse_args()
    d = load(a.source)
    if a.list:
        with pd.option_context("display.width", 250, "display.max_colwidth", 70):
            print(list_brands(d).to_string(index=False))
    elif a.brand:
        w = weekly(d, a.brand)
        w.to_csv(a.out or f"{a.brand}.csv", index=False)
        print(f"{len(w)} weeks, channels: {[c for c in w if c.startswith('spend_')]} -> {a.out}")


if __name__ == "__main__":
    main()
