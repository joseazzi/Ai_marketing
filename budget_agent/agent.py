"""The LLM agent: plans, calls the model/optimizer as tools, acts, and verifies its own numbers.

Loop per user message:
 1. The LLM writes a plan (``make_plan``) - e.g. check data -> fit -> compare -> recommend.
 2. It executes the plan by calling tools; results go back into the conversation.
 3. It can act: write a budget plan file, send saturation alerts, ingest new data.
 4. Before the answer is returned, a grounding check verifies every figure in it
    appears in a tool result. If not, the agent is asked to correct itself once.
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

from . import grounding
from .tools import dispatch, openai_tools, to_json
from .workspace import Workspace

SYSTEM_PROMPT = """You are the Budget Agent, an AI marketing analyst that decides how to split a marketing budget
across channels using a Media Mix Model (MMM) and an optimizer, which you run as tools.

How you work:
1. For every new goal, FIRST call make_plan with concrete ordered steps. A typical plan:
   check_data -> fit_model (if no model or data changed) -> get_channel_performance -> optimize_budget
   -> run_what_if / compare_scenarios if useful -> check_saturation_alerts -> generate_budget_plan -> answer.
   Skip steps that are clearly unnecessary (e.g. don't refit if the model is current), but say why.
2. Execute the plan with tools. If a tool returns an error, adapt the plan (e.g. fit the model first).
3. Act, don't only answer: when the user wants a plan, call generate_budget_plan and give the file path.
   When channels are saturated, call check_saturation_alerts so the team is notified.
   When new weekly results are provided, call ingest_new_data and explain what changed and why.
4. Interpret budgets: "$100K for Q1" = total_budget 100000, n_weeks 13. "a month" = 4 weeks, "a year" = 52.
5. Answer in plain business language: the recommended split, the expected effect versus the current split,
   and the REASON for each big move (saturation, marginal return of the next dollar, carry-over).
6. TRUST RULE: every number you state must come from a tool result in this conversation. Never invent or
   estimate figures yourself. Round sensibly ($23.4K, 15%). If the model fit is weak, say so.
7. PROFIT: revenue ROI above 1 does not mean an ad pays off. If the user gives a gross margin, call set_gross_margin
   and use goal="max_profit" when they care about profit. If no margin is set, say figures are revenue, not profit.
8. REALISM: if a plan without limits sets a channel to $0 or more than doubles one, say it is an aggressive plan
   and offer the version with max_change_pct=0.3 (each channel within +/-30% of today).
9. CHANGE LIMITS: with goal="max_sales", max_change_pct keeps every channel within +/-that %. With goal="max_profit"
   it only caps increases; cuts are always allowed so no channel is forced to lose money. A channel cut by more than
   max_change_pct in profit mode is therefore expected, not an inconsistency: explain it with change_limit_rule.
10. EXPLAINING A PLAN: to explain a plan's moves, use only optimize_budget fields: biggest_moves (every channel that
   changed by more than 5%, with direction, current and recommended allocation, reason), allocation,
   current_split.marginal_roi and recommended.marginal_roi. Read the direction from the data; never guess it.
   get_channel_performance describes HISTORICAL performance: its marginal_roi_now is at recent historical spend,
   not at the plan's spend. Use it only to describe the past, and say so when you quote it. Never mix the two
   figures, and compare a next-dollar return with break-even only within the same source.
11. ONLY REAL OPTIONS: only offer next actions the available tools can actually perform. Never promise an option a
   tool does not support. For example, there is no way to apply a change limit to cuts in profit mode; to keep every
   channel within +/-max_change_pct, the option is goal="max_sales" with that max_change_pct.
   Every next action you offer must map to one tool call, described by what it does, e.g. "re-run with
   max_change_pct=0.3" (optimize_budget), "optimise for profit" (optimize_budget goal="max_profit"), "run a
   what-if" (run_what_if), "compare scenarios" (compare_scenarios), "write the plan file" (generate_budget_plan),
   "check saturation and alert the team" (check_saturation_alerts), "ingest new weeks" (ingest_new_data),
   "set the gross margin" (set_gross_margin). Never offer or promise scheduling, handoff, deployment, launching
   campaigns, contacting people, or anything else that is not in the tool list.
12. PROFIT NOTE: if optimize_budget returns profit_note, always state it in one sentence (the plan's last dollars
   lose money at this margin, and what goal="max_profit" would give instead).
13. HOW THE PLAN IS DECIDED: optimize_budget maximises the model's fitted response curves (sales as a function of
   spend for each channel) under the budget and limits. That is the only thing that decides the plan.
   get_channel_performance and check_saturation_alerts only describe and report; they never feed the optimiser.
   Never say alerts or historical channel figures were "used in" or "drove" the optimisation. Say the plan comes
   from the model's response curves, and use historical figures only as context, labelled as historical.
14. "BASED ON WHAT?": when asked what a recommendation is based on (or how reliable it is), state:
   (a) the data period and number of weeks (fit_model data_period or check_data start/end/weeks);
   (b) model accuracy from the tools: in-sample R2 and the holdout R2 and MAPE from fit_model holdout_accuracy
   (accuracy on the last 13 weeks when the model was trained without them);
   (c) that channel effects are estimated from past week-to-week variation, not measured by experiments, and that
   channels whose spend moves together are hard to separate (quote channels_hard_to_separate if not empty);
   (d) if is_simulated_demo is true, that the data is the simulated demo company, not real company data.
   Call fit_model or check_data first if you don't have these fields in this conversation.
15. COPY FIGURES EXACTLY: copy numbers from tool results. Round only by standard rounding of the tool value
   (roi 1.505 -> "1.51" or "1.5"; 1.263 -> "1.26" or "1.3"), never re-estimate or write a nearby value. You may
   repeat the user's own figures (their budget, minimums) exactly as they gave them.
16. WHAT-IF ANSWERS: start with run_what_if's verdict ("better", "worse" or "about the same", on verdict_metric:
   profit when a margin is set, else sales) and the change (verdict_change). Then give the reason: channels pushed
   below break-even (channels_below_breakeven), or money moved to channels with a lower / higher next-dollar
   return (channels_changed: base_marginal_roi -> new_marginal_roi). If base_note is present (the base was an
   optimised plan), say the change cannot beat it. Never call a worse result "nearly neutral" without the reason.
17. WORDING: optimize_budget's marginal_roi values (current_split, recommended, biggest_moves) are figures for the
   plan's own spend; never call them "historical". Always write sales and profit amounts with $
   (e.g. "$75,885 incremental sales").
18. Be concise: a short summary, a compact table of the split, 2-4 reasons, next actions.
"""


@dataclass
class AgentStep:
    tool: str
    args: dict
    result: dict
    seconds: float


@dataclass
class AgentResponse:
    answer: str
    plan: list[dict] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    revised: bool = False


def _strip_meta(text: str) -> str:
    """Drop a leading line like 'Rewritten using only numbers from tool outputs.' if the model adds one."""
    lines = text.lstrip().splitlines()
    if lines and re.match(r"^\W*(rewritten|revised|corrected|updated answer|here is the (rewritten|revised|corrected))",
                          lines[0], re.I):
        lines = lines[1:]
    return "\n".join(lines).lstrip()


class BudgetAgent:
    def __init__(self, workspace: Workspace, model: str | None = None, api_key: str | None = None,
                 max_tool_rounds: int = 15, client=None, tool_lock=None, max_turns_kept: int = 6,
                 old_tool_chars: int = 1500):
        self.ws = workspace
        # tool_lock guards the shared workspace while a tool runs (not while the LLM thinks);
        # chat_lock serialises conversations on this agent so its message history stays consistent.
        self.tool_lock = tool_lock or contextlib.nullcontext()
        self.chat_lock = threading.Lock()
        self.max_turns_kept = max_turns_kept
        self.old_tool_chars = old_tool_chars
        # full (never truncated) user messages and tool results of the kept turns, for the number check
        self.grounding_turns: list[dict] = []
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-5-mini")
        self.max_tool_rounds = max_tool_rounds
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.client = client
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _context_note(self) -> str:
        ws = self.ws
        parts = []
        if ws.df is None:
            parts.append("No data loaded yet.")
        else:
            parts.append(f"Data loaded: {len(ws.df)} weeks, {ws.df['date'].min().date()} to {ws.df['date'].max().date()}.")
            parts.append("Model: " + ("trained on %d weeks." % ws.model.trained_weeks if ws.model else "not trained."))
            if ws.model and ws.model.trained_weeks != len(ws.df):
                parts.append("Data has changed since the last fit: refit before recommending.")
        parts.append(f"Gross margin: {ws.gross_margin:.0%}." if ws.gross_margin else "Gross margin: not set (revenue mode).")
        if ws.scenarios:
            parts.append(f"Stored scenarios: {list(ws.scenarios)}.")
        if ws.demo_weeks_remaining():
            parts.append(f"Demo stream has {ws.demo_weeks_remaining()} new weeks available (ingest_new_data demo_weeks).")
        return "[workspace state] " + " ".join(parts)

    def _complete(self):
        return self.client.chat.completions.create(model=self.model, messages=self.messages, tools=openai_tools())

    def _compact_history(self) -> None:
        """Keep the conversation bounded: one current state note, short old tool outputs, last N turns."""
        msgs = [self.messages[0]] + [m for m in self.messages[1:]
                                     if not (m["role"] == "system" and m["content"].startswith("[workspace state]"))]
        for m in msgs:
            if m["role"] == "tool" and len(m["content"]) > self.old_tool_chars:
                m["content"] = m["content"][: self.old_tool_chars] + ' ... [truncated: older result]"'
        turns = [i for i, m in enumerate(msgs) if m["role"] == "user" and not m["content"].startswith("[internal check")]
        keep = max(self.max_turns_kept - 1, 0)  # previous turns kept; the new question makes it max_turns_kept
        if len(turns) > keep:
            msgs = [msgs[0]] + (msgs[turns[-keep]:] if keep else [])
        self.messages = msgs

    def ask(self, user_message: str, on_step=None) -> AgentResponse:
        with self.chat_lock:
            return self._ask(user_message, on_step)

    def _grounding_pool(self, current_results: list) -> tuple[list, list[str]]:
        """Everything the agent may quote: tool results of the kept turns (not truncated), stored scenarios,
        the last saved recommendation, and the user's own messages."""
        results = [r for t in self.grounding_turns for r in t["results"]] + list(current_results)
        with self.tool_lock:
            results += [p.to_dict() for p in getattr(self.ws, "scenarios", {}).values()]
            recs = getattr(self.ws, "memory", {}).get("recommendations") or []
            if recs:
                results.append(recs[-1])
        return results, [t["user"] for t in self.grounding_turns]

    def _ask(self, user_message: str, on_step=None) -> AgentResponse:
        self._compact_history()
        keep = max(self.max_turns_kept - 1, 0)
        self.grounding_turns = self.grounding_turns[-keep:] if keep else []
        self.grounding_turns.append({"user": user_message, "results": []})
        with self.tool_lock:
            note = self._context_note()
        self.messages.append({"role": "system", "content": note})
        self.messages.append({"role": "user", "content": user_message})
        plan: list[dict] = []
        steps: list[AgentStep] = []
        tool_results: list = []

        def run_loop() -> str:
            for _ in range(self.max_tool_rounds):
                msg = self._complete().choices[0].message
                if not msg.tool_calls:
                    self.messages.append({"role": "assistant", "content": msg.content or ""})
                    return msg.content or ""
                self.messages.append({
                    "role": "assistant", "content": msg.content or "",
                    "tool_calls": [{"id": tc.id, "type": "function",
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                   for tc in msg.tool_calls],
                })
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    t0 = time.time()
                    with self.tool_lock:
                        result = dispatch(self.ws, tc.function.name, args, plan_sink=plan)
                    step = AgentStep(tc.function.name, args, result, time.time() - t0)
                    steps.append(step)
                    tool_results.append(result)
                    self.grounding_turns[-1]["results"].append(result)
                    if on_step:
                        on_step(step)
                    self.messages.append({"role": "tool", "tool_call_id": tc.id, "content": to_json(result)})
            return "I stopped after too many tool calls. Here is what I found so far: " + to_json(
                [s.tool for s in steps])

        answer = run_loop()
        pool, users = self._grounding_pool([])
        check = grounding.check(answer, pool, user_texts=users)
        revised = False
        if not check["passed"]:
            self.messages.append({"role": "user", "content": (
                "[internal check, invisible to the user] These figures in your last answer do not match any tool "
                f"result or in the user's messages: {check['ungrounded']}. Write the answer again for the user: fix or remove only those "
                "figures (call tools again if you need a number), keep the same structure, plain business language "
                "and sensible rounding ($23.4K, 15%), and never show raw field names. Start directly with the answer: "
                "do not say it was rewritten, checked or corrected.")})
            answer = _strip_meta(run_loop())
            pool, users = self._grounding_pool([])
            check = grounding.check(answer, pool, user_texts=users)
            revised = True
        return AgentResponse(answer, plan, steps, check, revised)

    def reset(self) -> None:
        with self.chat_lock:  # waits for an answer in progress instead of mixing histories
            self.messages = self.messages[:1]
            self.grounding_turns = []
