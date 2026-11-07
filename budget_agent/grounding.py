"""Trust check: every number the agent states must come from a tool result.

Extracts figures like "$23,400", "15.2%", "2.31", "$100K" from the answer and checks
each one against the numbers that appeared in tool outputs this turn. A figure
counts as grounded only if it equals a tool value, allowing for:

* the rounding the writer actually used ("$23.4K" may differ by up to $50,
  "$23K" by up to $500, "2.31" by up to 0.005);
* fractions written as percentages (0.152 -> "15.2%");
* totals written per week (value / n_weeks, when a tool reported n_weeks);
* sign ("-$1,197" matches a change of -1197.3 or 1197.3);
* English or French number formats: "84,186.52", "84 186,52 $" (space, no-break or narrow no-break space as
  thousands separator), "0,462", "15,8K". A genuinely ambiguous "1,500" is accepted if either 1500 or 1.5 matches.

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

_SPACES = " \u00a0\u202f"
NUM_RE = re.compile(
    r"(?<![\w.,])(\$)?\s?"
    r"(-?(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?!\d)(?:,\d+)?"   # French: 84 186,52 (space-grouped thousands)
    r"|\d+(?:,\d+)*(?:\.\d+)?))"                                # English 84,186.52 / plain 0,462 / 1500
    r"[ \u00a0\u202f]?(%|[kKmM](?![a-zA-Z]))?"
    r"([ \u00a0\u202f]?\$)?")
# plain integers followed by these words are counts or periods, not model figures
_UNIT_AFTER = re.compile(r"^\s*-?\s*(weeks?|wks?|months?|days?|years?|steps?|channels?|quarters?|scenarios?|"
                         r"figures?|alerts?|reasons?|actions?|options?|semaines?|mois|jours?)\b", re.I)
_REL_TOL = 1e-6  # float noise only; rounding is handled by the claim's own precision


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
            vals = _interpretations(m.group(2), "")
            if vals:
                out.append(vals[0][0])


def _interpretations(token: str, suffix: str, english_only: bool = False) -> list[tuple[float, float]]:
    """All plausible (value, precision) readings of a number token, English or French format.

    precision = half a unit of the last digit the writer used ("$23K" -> 500, "2.31" -> 0.005)."""
    neg = token.startswith("-")
    t = token.lstrip("-")
    mult = {"k": 1_000, "m": 1_000_000}.get(suffix, 1)
    readings: list[tuple[str, str]] = []                 # (integer digits, decimal digits)
    if any(c in t for c in _SPACES):                      # 84 186,52 -> French grouping, comma decimals
        whole, _, dec = "".join(c for c in t if c not in _SPACES).partition(",")
        readings.append((whole, dec))
    elif "." in t:                                        # 84,186.52 -> English
        whole, _, dec = t.partition(".")
        readings.append((whole.replace(",", ""), dec))
    elif "," in t:
        parts = t.split(",")
        english = len(parts[0]) <= 3 and parts[0].strip("0") != "" and all(len(p) == 3 for p in parts[1:])
        if english:
            readings.append(("".join(parts), ""))         # 1,500 -> 1500
        if len(parts) == 2 and not (english and english_only):
            readings.append((parts[0], parts[1]))         # 0,462 -> 0.462 ; 1,500 may also be 1.5
    else:
        readings.append((t, ""))
    out = []
    for whole, dec in readings:
        if not whole.isdigit() or (dec and not dec.isdigit()):
            continue
        v = float(f"{whole}.{dec}" if dec else whole) * mult * (-1 if neg else 1)
        if dec:
            prec = 0.5 * 10 ** -len(dec) * mult
        else:
            zeros = len(whole) - len(whole.rstrip("0")) if whole.strip("0") else 0
            prec = 0.5 * 10 ** zeros * mult
        out.append((v, prec))
    return out


def extract_claims(text: str) -> list[tuple[str, list[tuple[float, float]], str]]:
    """Figures in the text as (raw text, [(value, precision), ...], suffix)."""
    claims = []
    for m in NUM_RE.finditer(text):
        raw = m.group(0).strip()
        suffix = (m.group(3) or "").lower()
        # a leading "$" is English style, so "$1,500" is 1500 only; French writes "1 500 $"
        readings = _interpretations(m.group(2), suffix, english_only=bool(m.group(1)))
        if not readings:
            continue
        v = readings[0][0]
        money = bool(m.group(1) or m.group(4))
        if not money and suffix not in ("%", "k", "m") and len(readings) == 1 and v == int(v):
            before = text[max(0, m.start() - 2):m.start()]
            line_start = text.rfind("\n", 0, m.start()) + 1
            if (abs(v) <= 12                                             # counts / option numbers: "1, 2, 3"
                    or 1990 <= v <= 2100                                 # years
                    or _UNIT_AFTER.match(text[m.end():m.end() + 20])     # "13 weeks", "3 alerts"
                    or before.endswith(("Q", "q", "#"))                  # Q1, #2
                    or (text[line_start:m.start()].strip(" *-") == ""    # list markers "1." / "2)"
                        and text[m.end():m.end() + 1] in (".", ")"))
                    or re.search(r"\d{4}-$", text[max(0, m.start() - 5):m.start()])):  # dates 2026-01-05
                continue
        claims.append((raw, readings, suffix))
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
    for raw, readings, suffix in extract_claims(answer):
        targets = list(readings) + ([(v / 100, p / 100) for v, p in readings] if suffix == "%" else [])
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
