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
# numbers that are parameters, not claims: `code spans`, dict literals {"tv": -0.1},
# and (...) / {...} right after a snake_case tool or field name ("run_what_if(changes={...})")
_CODE_SPAN = re.compile(r"`[^`\n]*`")
_DICT_LIT = re.compile(r"\{[^{}\n]*:[^{}\n]*\}")
_TOOL_ARGS = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b[^\n({]{0,30}?[({][^(){}\n]*[)}]")
_USER_WEEKS = re.compile(r"(\d+)\s*(?:weeks?|wks?|semaines?)\b", re.I)
_QUARTER = re.compile(r"\b(?:q[1-4]|quarter|trimestre)\b", re.I)
_RATIO_MAX = 20  # plain numbers below this (no $, no %) are ratios: compared with the written precision only

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


def _mask_parameters(text: str) -> str:
    """Blank out parameter examples so their numbers are not checked as figures (positions are kept)."""
    for rx in (_CODE_SPAN, _DICT_LIT, _TOOL_ARGS):
        text = rx.sub(lambda m: " " * len(m.group(0)), text)
    return text


def extract_claims(text: str, skip_counts: bool = True) -> list[tuple[str, list[tuple[float, float]], str, bool]]:
    """Figures in the text as (raw text, [(value, precision), ...], suffix, is_money)."""
    claims = []
    text = _mask_parameters(text)
    for m in NUM_RE.finditer(text):
        raw = m.group(0).strip()
        suffix = (m.group(3) or "").lower()
        # a leading "$" is English style, so "$1,500" is 1500 only; French writes "1 500 $"
        readings = _interpretations(m.group(2), suffix, english_only=bool(m.group(1)))
        if not readings:
            continue
        v = readings[0][0]
        money = bool(m.group(1) or m.group(4))
        if skip_counts and not money and suffix not in ("%", "k", "m") and len(readings) == 1 and v == int(v):
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
        claims.append((raw, readings, suffix, money))
    return claims


def _close(claim: float, value: float, precision: float, rel_tol: float = _REL_TOL) -> bool:
    # tiny epsilon so a value exactly half a unit away (1.505 written as 1.51) is not lost to float error
    return abs(abs(claim) - abs(value)) <= max(precision * (1 + 1e-9), abs(value) * rel_tol, 1e-9)


def _user_numbers(texts: list[str]) -> tuple[list[float], set[int]]:
    """Figures the user wrote (exact values, K/M/% applied) and any period they stated in weeks."""
    nums, weeks = [], set()
    for t in texts or []:
        for _, readings, suffix, _ in extract_claims(t, skip_counts=False):
            for v, _ in readings:
                nums.append(v)
                if suffix == "%":
                    nums.append(v / 100)
        weeks.update(int(w) for w in _USER_WEEKS.findall(t) if 0 < int(w) <= 520)
        if _QUARTER.search(t):
            weeks.add(13)
    return nums, weeks


def check(answer: str, tool_results: list, user_texts: list[str] | None = None) -> dict:
    """``tool_results``: every tool output the agent may quote (this and earlier turns, stored scenarios).
    ``user_texts``: the user's messages in the conversation; their own figures may be repeated."""
    nums: list[float] = []
    weeks: set[int] = set()
    for r in tool_results:
        if isinstance(r, str) and r.startswith(("{", "[")):
            r = json.loads(r)
        _numbers_in(r, nums, weeks)
    unums, uweeks = _user_numbers(user_texts)
    nums += unums
    weeks |= uweeks
    pool = sorted({round(n, 6) for n in nums})
    candidates = set(pool)
    candidates.update(n * 100 for n in pool)                       # fraction -> percent
    for w in weeks:
        candidates.update(n / w for n in pool if abs(n) >= w)      # period total -> per week
    cand = sorted(candidates)

    grounded, ungrounded = [], []
    for raw, readings, suffix, money in extract_claims(answer):
        targets = list(readings) + ([(v / 100, p / 100) for v, p in readings] if suffix == "%" else [])
        if not money and suffix == "" and all(abs(v) < _RATIO_MAX for v, _ in readings):
            # a ratio (ROI, next-$ return): must be a standard rounding of a raw tool value, nothing derived
            ok = any(_close(t, c, p, rel_tol=0) for t, p in targets for c in pool)
        else:
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
