"""Trust check: every number the agent states must come from a tool result.

Extracts figures like "$23,400", "15.2%", "2.31", "$100K" from the answer and checks
each one against the numbers that appeared in tool outputs this turn. A figure
counts as grounded only if it equals a tool value, allowing for:

* the rounding the writer actually used ("$23.4K" may differ by up to $50,
  "$23K" by up to $500, "2.31" by up to 0.005);
* fractions written as percentages (0.152 -> "15.2%");
* totals written per week (value / n_weeks, when a tool reported n_weeks);
* sign ("-$1,197" matches a change of -1197.3 or 1197.3).

It deliberately does NOT accept sums or differences of arbitrary pairs: with a few
hundred tool values that makes almost any invented number "match". Figures the
agent computes itself (e.g. a difference no tool returned) are therefore flagged,
and the agent is asked to use a tool value instead.

What this guarantees: numbers are copied from the model, not invented. What it
doesn't: that the agent's reasoning about those numbers is right.
"""
from __future__ import annotations

import json
import re

NUM_RE = re.compile(r"(?<![\w.])(\$)?\s?(-?\d[\d,]*(?:\.\d+)?)\s?(%|[kKmM](?![a-zA-Z]))?")
# plain integers followed by these words are counts or periods, not model figures
_UNIT_AFTER = re.compile(r"^\s*-?\s*(weeks?|wks?|months?|days?|years?|steps?|channels?|quarters?|scenarios?|"
                         r"figures?|alerts?|reasons?|actions?|options?|semaines?|mois|jours?)\b", re.I)
_REL_TOL = 1e-3  # float noise only; rounding is handled by the claim's own precision


def _numbers_in(obj, out: list[float], weeks: set[int]) -> None:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("n_weeks", "weeks") and isinstance(v, int) and 1 <= v <= 520:
                weeks.add(v)
            _numbers_in(v, out, weeks)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _numbers_in(v, out, weeks)
    elif isinstance(obj, str):
        for m in NUM_RE.finditer(obj):
            try:
                out.append(float(m.group(2).replace(",", "")))
            except ValueError:
                pass


def _precision(digits: str, suffix: str) -> float:
    """Half a unit of the last significant digit the writer used ("$23K" -> 500, "2.31" -> 0.005)."""
    digits = digits.replace(",", "").lstrip("-")
    mult = {"k": 1_000, "m": 1_000_000}.get(suffix, 1)
    if "." in digits:
        return 0.5 * 10 ** -len(digits.split(".")[1]) * mult
    zeros = len(digits) - len(digits.rstrip("0")) if digits.strip("0") else 0
    return 0.5 * 10 ** zeros * mult


def extract_claims(text: str) -> list[tuple[str, float, str, float]]:
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
        if not m.group(1) and suffix not in ("%", "k", "m") and v == int(v):
            before = text[max(0, m.start() - 2):m.start()]
            line_start = text.rfind("\n", 0, m.start()) + 1
            if (1990 <= v <= 2100                                        # years
                    or _UNIT_AFTER.match(text[m.end():m.end() + 20])     # "13 weeks", "3 alerts"
                    or before.endswith(("Q", "q", "#"))                  # Q1, #2
                    or (text[line_start:m.start()].strip(" *-") == ""    # list markers "1." / "2)"
                        and text[m.end():m.end() + 1] in (".", ")"))
                    or re.search(r"\d{4}-$", text[max(0, m.start() - 5):m.start()])):  # dates 2026-01-05
                continue
        claims.append((raw, v, suffix, _precision(m.group(2), suffix)))
    return claims


def _close(claim: float, value: float, precision: float) -> bool:
    return abs(abs(claim) - abs(value)) <= max(precision, abs(value) * _REL_TOL, 1e-9)


def check(answer: str, tool_results: list) -> dict:
    nums: list[float] = []
    weeks: set[int] = set()
    for r in tool_results:
        if isinstance(r, str) and r.startswith(("{", "[")):
            r = json.loads(r)
        _numbers_in(r, nums, weeks)
    pool = sorted({round(n, 6) for n in nums})
    candidates = set(pool)
    candidates.update(n * 100 for n in pool)                       # fraction -> percent
    for w in weeks:
        candidates.update(n / w for n in pool if abs(n) >= w)      # period total -> per week
    cand = sorted(candidates)

    grounded, ungrounded = [], []
    for raw, v, suffix, prec in extract_claims(answer):
        targets = [(v, prec)] + ([(v / 100, prec / 100)] if suffix == "%" else [])
        ok = any(_close(t, c, p) for t, p in targets for c in cand)
        (grounded if ok else ungrounded).append(raw)
    total = len(grounded) + len(ungrounded)
    return {
        "numbers_checked": total,
        "grounded": grounded,
        "ungrounded": ungrounded,
        "score": 1.0 if total == 0 else len(grounded) / total,
        "passed": not ungrounded,
    }
