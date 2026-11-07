"""Trust check: every number the agent states must come from a tool result.

Extracts figures like "$23,400", "15.2%", "2.3x", "$100K" from the answer and
checks each against all numbers that appeared in tool outputs (allowing for
rounding, percentages written as fractions, K/M suffixes and simple sums of
two values). Anything unmatched is flagged so the agent can correct itself.
"""
from __future__ import annotations

import json
import re

NUM_RE = re.compile(r"(?<![\w.])(\$)?\s?(-?\d[\d,]*(?:\.\d+)?)\s?(%|[kKmM](?![a-zA-Z]))?")


def _numbers_in(obj, out: list[float]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            _numbers_in(v, out)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, out)
    elif isinstance(obj, str):
        for m in NUM_RE.finditer(obj):
            try:
                out.append(float(m.group(2).replace(",", "")))
            except ValueError:
                pass


def extract_claims(text: str) -> list[tuple[str, float, str]]:
    claims = []
    for m in NUM_RE.finditer(text):
        raw = m.group(0).strip()
        try:
            v = float(m.group(2).replace(",", ""))
        except ValueError:
            continue
        suffix = (m.group(3) or "").lower()
        if suffix == "k":
            v *= 1_000
        elif suffix == "m":
            v *= 1_000_000
        # ignore small counts / list markers / years: weeks, steps, "Q1", 2026
        if suffix != "%" and not m.group(1) and (abs(v) <= 60 and v == int(v) or 1990 <= v <= 2100):
            continue
        claims.append((raw, v, suffix, _precision(m.group(2), suffix)))
    return claims


def _precision(digits: str, suffix: str) -> float:
    """Half a unit of the last significant digit the writer used ("$23K" -> 500)."""
    digits = digits.replace(",", "").lstrip("-")
    mult = {"k": 1_000, "m": 1_000_000}.get(suffix, 1)
    if "." in digits:
        return 0.5 * 10 ** -len(digits.split(".")[1]) * mult
    zeros = len(digits) - len(digits.rstrip("0")) if digits.strip("0") else 0
    return 0.5 * 10 ** zeros * mult


def _close(claim: float, value: float, precision: float = 0.0) -> bool:
    tol = max(abs(value) * 0.01, precision, 0.006 if abs(value) < 10 else 0.5)
    return abs(abs(claim) - abs(value)) <= tol


def check(answer: str, tool_results: list) -> dict:
    nums: list[float] = []
    for r in tool_results:
        _numbers_in(r if not isinstance(r, str) else json.loads(r) if r.startswith(("{", "[")) else r, nums)
    pool = set(round(n, 4) for n in nums)
    pool_list = list(pool)
    candidates = set(pool)
    for n in pool_list:
        candidates.update({n * 100, n / 1000, n / 1_000_000})
    # differences/sums of pairs (e.g. "+$12,000 vs current") - bounded to keep it fast
    big = sorted({n for n in pool_list if abs(n) >= 1}, key=abs, reverse=True)[:60]
    pair = {a - b for a in big for b in big} | {a + b for a in big for b in big}

    grounded, ungrounded = [], []
    for raw, v, suffix, prec in extract_claims(answer):
        targets = [(v, prec)]
        if suffix == "%":
            targets += [(v / 100, prec / 100)]
        ok = (any(_close(t, c, pr) for t, pr in targets for c in candidates)
              or any(_close(v, p, prec) for p in pair))
        (grounded if ok else ungrounded).append(raw)
    total = len(grounded) + len(ungrounded)
    return {
        "numbers_checked": total,
        "grounded": grounded,
        "ungrounded": ungrounded,
        "score": 1.0 if total == 0 else len(grounded) / total,
        "passed": not ungrounded,
    }
