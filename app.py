"""Streamlit dashboard + chat for the AI Marketing Budget Allocation Optimizer.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from budget_agent import data as D
from budget_agent.env import load_dotenv
from budget_agent.workspace import Workspace

load_dotenv()
st.set_page_config(page_title="AI Budget Allocation Agent", page_icon="📈", layout="wide")

# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
if "ws" not in st.session_state:
    st.session_state.ws = Workspace(os.getenv("BUDGET_AGENT_WORKSPACE", "workspace"))
    st.session_state.chat = []
    st.session_state.agent = None
ws: Workspace = st.session_state.ws


def money(x: float) -> str:
    return f"${x:,.0f}"


def get_agent():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    if st.session_state.agent is None:
        from budget_agent.agent import BudgetAgent
        st.session_state.agent = BudgetAgent(ws)
    return st.session_state.agent


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("📈 Budget Agent")
    st.caption("Media Mix Model + optimizer + AI agent")

    st.subheader("1 · Data")
    src = st.radio("Source", ["Demo company (simulated)", "Upload CSV"], label_visibility="collapsed")
    if src.startswith("Demo"):
        seed = st.number_input("Seed", 1, 9999, 7)
        if st.button("Load demo data", use_container_width=True):
            ws.init_demo(seed=int(seed))
            st.session_state.agent = None
            st.success("Loaded 156 weeks of simulated data.")
    else:
        up = st.file_uploader("Weekly CSV", type="csv")
        st.caption("Columns: `date, sales, spend_<channel>…` + optional controls (holiday, promo…)")
        if up is not None and st.button("Use this file", use_container_width=True):
            try:
                ws.set_data(D.load_csv(up), source=up.name)
                st.session_state.agent = None
                st.success("Data loaded.")
            except Exception as e:
                st.error(str(e))
        tmpl = D.simulate(weeks=8).data.to_csv(index=False)
        st.download_button("Download CSV template", tmpl, "template.csv", use_container_width=True)

    st.subheader("2 · Profit")
    gm = st.number_input("Gross margin (%)", 0, 100, int(round((ws.gross_margin or 0) * 100)),
                         help="Share of each sale kept after product costs. 0 = unknown: results stay in revenue.")
    if (gm / 100 or None) != ws.gross_margin:
        ws.set_gross_margin(gm / 100 if gm else None)
    st.caption(f"Break-even: ${ws.breakeven:.2f} of sales per $1 of ads" if ws.gross_margin
               else "No margin set: results are revenue, not profit.")

    st.subheader("3 · OpenAI")
    if not os.getenv("OPENAI_API_KEY"):
        k = st.text_input("API key", type="password", help="Or set OPENAI_API_KEY in .env")
        if k:
            os.environ["OPENAI_API_KEY"] = k
            st.rerun()
    else:
        st.success(f"Key set · model `{os.getenv('OPENAI_MODEL', 'gpt-5-mini')}`")

    st.divider()
    if ws.df is not None:
        st.caption(f"Data: {len(ws.df)} weeks · {ws.df['date'].min().date()} → {ws.df['date'].max().date()}")
        st.caption("Model: " + (f"trained on {ws.model.trained_weeks} weeks (R² {ws.model.fit_metrics['r2']:.2f})"
                                if ws.model else "not trained"))

if ws.df is None:
    st.title("AI Marketing Budget Allocation Optimizer")
    st.info("👈 Load the demo company or upload your weekly data to start.")
    st.stop()

tabs = st.tabs(["🤖 Agent", "📊 Data", "🧠 Model", "🎯 Optimizer", "🔀 What-if", "📡 Monitoring", "✅ Evaluation"])

# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
with tabs[0]:
    st.header("Ask the Budget Agent")
    agent = get_agent()
    if agent is None:
        st.warning("No OpenAI key: the chat is disabled, but the offline autopilot below runs the same "
                   "plan → model → optimize → alert → plan-file workflow.")
    col1, col2 = st.columns([3, 1])
    with col2:
        st.markdown("**Autopilot (no LLM)**")
        ap_budget = st.number_input("Budget ($)", 1000, 100_000_000, 100_000, step=5000, key="ap_b")
        ap_weeks = st.number_input("Weeks", 1, 52, 13, key="ap_w")
        if st.button("Run autopilot", use_container_width=True):
            from budget_agent.autopilot import run_goal
            logs = []
            with st.spinner("Planning and executing…"):
                res = run_goal(ws, float(ap_budget), int(ap_weeks), log=logs.append)
            st.session_state.chat.append({"role": "assistant", "content": "```\n" + "\n".join(logs) + "\n```",
                                          "steps": [], "plan": None, "grounding": None})
            st.rerun()
        st.markdown("**Try asking**")
        for q in ["I have $100K for Q1. How should I spend it?",
                  "What if I cut TV by 30%?",
                  "Which channels are saturated? Alert the team.",
                  "New weekly results are in: ingest 4 demo weeks and tell me what changed.",
                  "Give me a plan for $250K over 26 weeks with at least $40K on TV and no channel changing more than 40%."]:
            if st.button(q, use_container_width=True, disabled=agent is None):
                st.session_state.pending = q

    with col1:
        for m in st.session_state.chat:
            with st.chat_message(m["role"]):
                if m.get("plan"):
                    with st.expander("🗺️ Plan", expanded=False):
                        for i, s in enumerate(m["plan"]["steps"], 1):
                            st.markdown(f"{i}. {s}")
                if m.get("steps"):
                    with st.expander(f"🛠️ {len(m['steps'])} tool calls", expanded=False):
                        for s in m["steps"]:
                            err = isinstance(s["result"], dict) and "error" in s["result"]
                            st.markdown(f"{'❌' if err else '✅'} `{s['tool']}` {json.dumps(s['args'])} · {s['seconds']:.1f}s")
                            st.json(s["result"], expanded=False)
                st.markdown(m["content"])
                if m.get("grounding"):
                    g = m["grounding"]
                    badge = "🟢" if g["passed"] else "🟠"
                    st.caption(f"{badge} Grounding: {len(g['grounded'])}/{g['numbers_checked']} figures verified "
                               f"against tool outputs" + (f" · unverified: {g['ungrounded']}" if g["ungrounded"] else "")
                               + (" · self-corrected" if m.get("revised") else ""))

        prompt = st.chat_input("e.g. I have $100K for Q1, how should I spend it?", disabled=agent is None)
        prompt = prompt or st.session_state.pop("pending", None)
        if prompt and agent is not None:
            st.session_state.chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                status = st.status("Thinking…", expanded=True)
                try:
                    r = agent.ask(prompt, on_step=lambda s: status.write(f"🛠️ `{s.tool}` ({s.seconds:.1f}s)"))
                    status.update(label="Done", state="complete", expanded=False)
                    st.session_state.chat.append({
                        "role": "assistant", "content": r.answer, "revised": r.revised,
                        "plan": r.plan[-1] if r.plan else None, "grounding": r.grounding,
                        "steps": [{"tool": s.tool, "args": s.args, "result": s.result, "seconds": s.seconds} for s in r.steps],
                    })
                except Exception as e:
                    status.update(label="Error", state="error")
                    st.session_state.chat.append({"role": "assistant", "content": f"⚠️ {type(e).__name__}: {e}"})
            st.rerun()
        if st.session_state.chat and st.button("Clear chat"):
            st.session_state.chat = []
            if agent:
                agent.reset()
            st.rerun()

    plans = sorted((ws.root / "plans").glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if plans:
        st.divider()
        st.subheader("📄 Generated budget plans")
        latest = plans[0]
        c1, c2 = st.columns(2)
        c1.download_button(f"Download {latest.name}", latest.read_bytes(), latest.name, use_container_width=True)
        md = latest.with_suffix(".md")
        if md.exists():
            c2.download_button(f"Download {md.name}", md.read_bytes(), md.name, use_container_width=True)
            with st.expander("Preview latest brief"):
                st.markdown(md.read_text())

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
with tabs[1]:
    q = ws.check_data()
    st.header("Data quality")
    c = st.columns(4)
    c[0].metric("Weeks", q["weeks"])
    c[1].metric("Channels", len(q["channels"]))
    c[2].metric("Total spend", money(q["total_spend"]))
    c[3].metric("Total sales", money(q["total_sales"]))
    for i in q["issues"]:
        st.error(i)
    for w in q["warnings"]:
        st.warning(w)
    if not q["issues"] and not q["warnings"]:
        st.success("No data quality problems found.")
    spend = ws.df.melt(id_vars="date", value_vars=[D.SPEND_PREFIX + c for c in q["channels"]],
                       var_name="channel", value_name="spend")
    spend["channel"] = spend["channel"].str.replace(D.SPEND_PREFIX, "")
    st.plotly_chart(px.area(spend, x="date", y="spend", color="channel", title="Weekly spend by channel"),
                    use_container_width=True)
    st.plotly_chart(px.line(ws.df, x="date", y="sales", title="Weekly sales"), use_container_width=True)
    with st.expander("Raw data"):
        st.dataframe(ws.df, use_container_width=True)

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
with tabs[2]:
    st.header("Media Mix Model")
    c1, c2 = st.columns([1, 3])
    hl = c1.number_input("Recency half-life (weeks, 0 = off)", 0, 520, int(ws.recency_halflife or 0))
    if c1.button("Train / retrain model", type="primary"):
        with st.spinner("Fitting adstock + saturation model…"):
            ws.fit_model(recency_halflife_weeks=hl or 0, from_scratch=True)
        st.rerun()
    if ws.model is None:
        st.info("Train the model to see channel contributions, ROI and saturation curves.")
    else:
        m = ws.model
        k = st.columns(3)
        k[0].metric("R²", f"{m.fit_metrics['r2']:.3f}")
        k[1].metric("MAPE", f"{m.fit_metrics['mape']:.1%}")
        k[2].metric("Trained on", f"{m.trained_weeks} weeks")
        pred = pd.DataFrame({"date": ws.df["date"], "actual": ws.df["sales"], "predicted": m.predict(ws.df)})
        st.plotly_chart(px.line(pred, x="date", y=["actual", "predicted"], title="Actual vs predicted sales"),
                        use_container_width=True)
        dec = m.decompose(ws.df)
        st.plotly_chart(px.area(dec.melt(id_vars="date", var_name="component", value_name="sales"),
                                x="date", y="sales", color="component", title="Sales decomposition: baseline + channels"),
                        use_container_width=True)
        s = m.channel_summary(ws.df)
        c1, c2 = st.columns(2)
        fig = go.Figure([go.Bar(name="ROI (historical)", x=s.channel, y=s.roi),
                         go.Bar(name="Marginal ROI (next $)", x=s.channel, y=s.marginal_roi)])
        fig.add_hline(y=1, line_dash="dot", annotation_text="break-even")
        fig.update_layout(barmode="group", title="Return per $ by channel")
        c1.plotly_chart(fig, use_container_width=True)
        c2.plotly_chart(px.bar(s, x="channel", y="saturation_pct", title="Saturation at current spend (share of max effect)",
                               range_y=[0, 1]), use_container_width=True)
        rc = m.response_curves()
        fig = px.line(rc, x="weekly_spend", y="weekly_sales", color="channel",
                      title="Response curves: weekly spend → weekly incremental sales")
        for ch in m.channels:
            cur = m.spend_stats[ch]["last13_mean"]
            fig.add_scatter(x=[cur], y=[m.params[ch].response(cur)], mode="markers", showlegend=False,
                            marker=dict(size=10, symbol="x"), name=f"{ch} now")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("✕ = current weekly spend. Where the curve flattens, extra spend is wasted (saturation).")
        st.dataframe(s.round(3), use_container_width=True)

# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #
with tabs[3]:
    st.header("Budget optimizer")
    if ws.model is None:
        st.info("Train the model first (🧠 Model tab).")
    else:
        c = st.columns(3)
        budget = c[0].number_input("Total budget ($)", 1000, 100_000_000, 100_000, step=5000)
        weeks = c[1].number_input("Period (weeks)", 1, 52, 13)
        mc = c[2].number_input("Max change per channel vs current (%, 0 = no limit)", 0, 500, 30)
        goal = st.radio("Goal", ["Maximize sales (spend the full budget)", "Maximize profit (budget is a cap)"],
                        horizontal=True, disabled=not ws.gross_margin,
                        help="Profit mode needs a gross margin in the sidebar.")
        cons = pd.DataFrame({"channel": ws.model.channels, "min_total": 0.0, "max_total": float(budget)})
        cons = st.data_editor(cons, disabled=["channel"], hide_index=True, use_container_width=True)
        if st.button("Optimize", type="primary"):
            try:
                st.session_state.opt = ws.optimize_budget(
                    float(budget), int(weeks),
                    {r.channel: r.min_total for r in cons.itertuples() if r.min_total > 0} or None,
                    {r.channel: r.max_total for r in cons.itertuples() if r.max_total < budget} or None,
                    mc / 100 if mc else None,
                    "max_profit" if goal.startswith("Maximize profit") and ws.gross_margin else "max_sales")
                if "error" in st.session_state.opt:
                    st.error(st.session_state.opt["error"])
                    st.session_state.opt = None
            except Exception as e:
                st.error(str(e))
        opt = st.session_state.get("opt")
        if opt:
            k = st.columns(3)
            k[0].metric("Expected incremental sales", money(opt["recommended"]["total_incremental_sales"]),
                        f"{opt['uplift_vs_current_pct']:+.1%} vs current split")
            k[1].metric("Media ROI", f"{opt['recommended']['roi']:.2f}", f"{opt['recommended']['roi'] - opt['current_split']['roi']:+.2f}")
            k[2].metric("Uplift", money(opt["uplift_vs_current"]))
            if opt.get("recommended_profit") is not None:
                k = st.columns(3)
                k[0].metric("Profit from ads (after margin)", money(opt["recommended_profit"]),
                            f"{opt['profit_uplift_vs_current']:+,.0f} vs current split")
                k[1].metric("Current split profit", money(opt["current_split_profit"]))
                k[2].metric("Budget left unspent", money(opt["unspent_budget"]))
            comp = pd.DataFrame({"current split": opt["current_split"]["allocation"],
                                 "recommended": opt["recommended"]["allocation"]}).reset_index(names="channel")
            st.plotly_chart(px.bar(comp.melt(id_vars="channel", var_name="plan", value_name="budget"),
                                   x="channel", y="budget", color="plan", barmode="group",
                                   title="Current vs recommended split"), use_container_width=True)
            st.subheader("Why")
            for mv in opt["biggest_moves"]:
                st.markdown(f"- **{mv['channel']}** {mv['change']:+,.0f}: {mv['reason']} "
                            f"(next-$ return {mv['current_marginal_roi']:.2f} → {mv['optimized_marginal_roi']:.2f})")
            if st.button("📄 Generate ready-to-use budget plan"):
                st.session_state.plan_files = ws.generate_budget_plan("recommended")
            if pf := st.session_state.get("plan_files"):
                c1, c2 = st.columns(2)
                c1.download_button("Weekly plan (CSV)", Path(pf["csv_path"]).read_bytes(), Path(pf["csv_path"]).name)
                c2.download_button("Brief (Markdown)", Path(pf["brief_path"]).read_bytes(), Path(pf["brief_path"]).name)

# --------------------------------------------------------------------------- #
# What-if
# --------------------------------------------------------------------------- #
with tabs[4]:
    st.header("What-if scenarios")
    if ws.model is None:
        st.info("Train the model first.")
    else:
        base_name = st.selectbox("Base plan", [n for n in ["current split", "recommended"] if n in ws.scenarios]
                                 or ["current split"])
        wb = st.number_input("Budget for base plan ($, used if no plan yet)", 1000, 100_000_000, 100_000, step=5000)
        cols = st.columns(len(ws.model.channels))
        changes = {}
        for col, ch in zip(cols, ws.model.channels):
            v = col.slider(ch, -100, 200, 0, step=5, format="%d%%")
            if v:
                changes[ch] = v / 100
        realloc = st.toggle("Reallocate freed money to other channels (keep total constant)")
        if st.button("Run scenario", type="primary", disabled=not changes):
            st.session_state.wi = ws.run_what_if(changes, base=base_name, reallocate=realloc,
                                                 total_budget=float(wb), n_weeks=13)
        if wi := st.session_state.get("wi"):
            k = st.columns(3)
            k[0].metric("Incremental sales", money(wi["result"]["total_incremental_sales"]),
                        f"{wi['sales_change_vs_base']:+,.0f} vs base")
            k[1].metric("Budget", money(wi["result"]["total_budget"]), f"{wi['budget_change_vs_base']:+,.0f}")
            k[2].metric("ROI", f"{wi['result']['roi']:.2f}")
            st.dataframe(pd.DataFrame(ws.compare_scenarios()["scenarios"]).drop(columns="allocation"),
                         use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# Monitoring / learning
# --------------------------------------------------------------------------- #
with tabs[5]:
    st.header("Monitoring & learning from new data")
    st.caption("Each week, new results are added; the agent checks how well its last forecast did, updates the "
               "model, re-optimizes the last goal, and alerts the team if the plan or a channel's saturation changed.")
    if ws.model is None:
        st.info("Train the model first.")
    else:
        c1, c2 = st.columns(2)
        rem = ws.demo_weeks_remaining()
        n_new = c1.number_input("Demo weeks to ingest", 1, max(rem, 1), min(4, max(rem, 1)), disabled=rem == 0)
        if c1.button(f"Ingest next demo weeks ({rem} left)", disabled=rem == 0):
            with st.spinner("Updating model and plan…"):
                st.session_state.ingest = ws.ingest_new_data(demo_weeks=int(n_new))
            st.rerun()
        upn = c2.file_uploader("…or upload new weekly results (CSV)", type="csv", key="newweeks")
        if upn is not None and c2.button("Ingest uploaded weeks"):
            with st.spinner("Updating model and plan…"):
                st.session_state.ingest = ws.ingest_new_data(new_rows=D.load_csv(upn))
            st.rerun()
        if res := st.session_state.get("ingest"):
            if "error" in res:
                st.error(res["error"])
            else:
                o = res["previous_model_out_of_sample"]
                k = st.columns(3)
                k[0].metric("Weeks added", res["weeks_added"])
                if o:
                    k[1].metric("Last model's forecast error on new weeks", f"{o['mape']:.1%}")
                k[2].metric("Updated R²", f"{res['refit']['r2']:.3f}")
                chg = pd.DataFrame(res["channel_changes"]).T
                st.dataframe(chg, use_container_width=True)
                if "recommendation_shift" in res:
                    st.plotly_chart(px.bar(x=list(res["recommendation_shift"]), y=list(res["recommendation_shift"].values()),
                                           labels={"x": "channel", "y": "change in recommended budget ($)"},
                                           title="How the new data changed the recommendation"), use_container_width=True)
                for a in res["alerts"]:
                    st.warning(a["message"])
        if st.button("Run saturation check & send alerts"):
            st.session_state.alerts = ws.check_saturation_alerts(notify=True)
        if al := st.session_state.get("alerts"):
            st.caption(f"Delivered via: {', '.join(al['delivered_via']) or 'none'}")
            for a in al["alerts"]:
                (st.info if a["severity"] == "info" else st.warning)(a["message"])

        hist = ws.get_history(limit=50)
        if hist["model_history"]:
            mh = pd.DataFrame([{"data_end": h["data_end"], **h["roi"]} for h in hist["model_history"]])
            st.plotly_chart(px.line(mh.melt(id_vars="data_end", var_name="channel", value_name="roi"),
                                    x="data_end", y="roi", color="channel", markers=True,
                                    title="How the agent's ROI estimates evolved"), use_container_width=True)
        with st.expander("Alert log"):
            st.dataframe(pd.DataFrame(hist["recent_alerts"]), use_container_width=True)
        with st.expander("Recommendation history"):
            st.json(hist["recent_recommendations"], expanded=False)

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
with tabs[6]:
    st.header("Evaluation: can we trust it?")
    st.markdown("1. **Accuracy** – hold out the last 13 weeks, predict them (R², MAPE).  \n"
                "2. **Business value** – current split vs optimized split for the same budget.  \n"
                "3. **Validation** – on simulated data with *known* channel effects, does the model recover them, "
                "and how much of the truly achievable uplift does our plan capture?")
    eb = st.number_input("Budget for business-value test ($)", 1000, 100_000_000, 100_000, step=5000, key="evb")
    if st.button("Run evaluation", type="primary"):
        from budget_agent.evaluation import business_value, ground_truth_recovery, holdout_accuracy
        with st.spinner("Running three evaluations (≈30s)…"):
            st.session_state.eval = {
                "acc": holdout_accuracy(ws.df),
                "bv": business_value(ws.model, float(eb), 13, df=ws.df) if ws.model else None,
                "gt": ground_truth_recovery(),
            }
    if ev := st.session_state.get("eval"):
        a = ev["acc"]
        st.subheader("1 · Accuracy (out-of-sample)")
        k = st.columns(4)
        k[0].metric("Train R²", f"{a['train_r2']:.3f}")
        k[1].metric("Train MAPE", f"{a['train_mape']:.1%}")
        k[2].metric("Test R²", f"{a['test_r2']:.3f}")
        k[3].metric("Test MAPE", f"{a['test_mape']:.1%}")
        st.plotly_chart(px.line(a["series"], x="date", y=["actual", "predicted"], title="Holdout forecast"),
                        use_container_width=True)
        if ev["bv"]:
            b = ev["bv"]
            st.subheader("2 · Business value")
            k = st.columns(3)
            k[0].metric("Current split", money(b["current_sales"]))
            k[1].metric("Optimized split", money(b["optimized_sales"]))
            k[2].metric("Expected uplift", money(b["uplift"]), f"{b['uplift_pct']:+.1%}")
        g = ev["gt"]
        st.subheader("3 · Validation against known truth (simulated)")
        k = st.columns(3)
        k[0].metric("ROI rank correlation", f"{g['roi_rank_correlation']:.2f}")
        k[1].metric("Mean ROI error", f"{g['mean_abs_pct_error']:.0%}")
        if g["share_of_best_uplift_captured"] is not None:
            k[2].metric("Share of best-possible uplift captured", f"{g['share_of_best_uplift_captured']:.0%}")
        gt = pd.DataFrame(g["channels"])
        st.plotly_chart(px.bar(gt.melt(id_vars="channel", value_vars=["true_roi", "estimated_roi"]),
                               x="channel", y="value", color="variable", barmode="group",
                               title="True vs estimated ROI"), use_container_width=True)
