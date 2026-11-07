# AI Marketing Budget Allocation Optimizer

An AI agent that decides how to split a marketing budget across channels so each dollar brings in as many sales as possible. It sits on top of a **Media Mix Model** (adstock + saturation) and a **SciPy optimizer**, and it does more than answer questions:

| What makes it an agent | Where |
|---|---|
| **Plans in steps by itself.** Before any work it writes a plan (`make_plan`): check the data, fit the model, compare scenarios, then recommend. If a step fails, it changes the plan. | `budget_agent/agent.py`, `autopilot.py` |
| **Acts.** It writes a ready-to-use budget plan (weekly spend per channel as CSV, plus a one-page brief) and sends **saturation alerts** to a log, Slack, a webhook or email. | `workspace.generate_budget_plan`, `check_saturation_alerts`, `notifier.py` |
| **Learns from new data.** When new weekly results arrive, it checks how accurate its last forecast was, updates the model (warm start plus a prior on the previous estimates), re-optimizes the last goal, reports what changed and raises an alert if the plan moved. | `workspace.ingest_new_data`, `autopilot.weekly_update` |
| **Checks its own numbers.** Every figure in the answer must match a tool result, allowing only for the rounding actually written, fractions shown as %, and per-week values. If one doesn't match, the agent rewrites the answer once. See [What the number check guarantees](#what-the-number-check-guarantees). | `grounding.py` |
| **Remembers.** Past recommendations, model versions, events and alerts persist in `workspace/memory.json`. | `workspace.py` |

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # put your OPENAI_API_KEY in it
python -m budget_agent.server # open http://localhost:8000
```

Go to **Data** → **Load demo**, then open **Agent** and ask: *"I have $100K for Q1. How should I spend it?"*

Without an API key, everything works except the free-form chat. The **Autopilot** card runs the same plan → model → optimize → alert → plan-file workflow without an LLM. You can also paste the key in **Settings**; it's kept in memory until the server stops.

### The web app (React)

Pages: **Agent** (chat with live plan, tool steps and a number check), **Data**, **Channel insights**, **Optimizer**, **What-if**, **Monitoring** and **Evaluation**. The built app (`frontend/dist`) is committed, so Node.js isn't needed to run it. To change the front end:

```bash
cd frontend && npm install
npm run dev        # http://localhost:5173, live reload; keep the Python server running for the API
npm run build      # rebuild frontend/dist
```

Stack: React 19, TypeScript, Vite, Tailwind CSS 4, Recharts and lucide icons. The API is FastAPI (`budget_agent/server.py`), and the chat streams each tool step as it happens.

The older Streamlit dashboard still works: `streamlit run app.py`.

### Command line

```bash
python -m budget_agent.cli demo                                   # simulated company, 3 years of weekly data
python -m budget_agent.cli load data/demo_company.csv             # ...or your own CSV
python -m budget_agent.cli plan --budget 100000 --weeks 13 --min tv=15000 --max-change 0.4
python -m budget_agent.cli weekly --demo-weeks 4                  # learning loop (or --csv new_weeks.csv)
python -m budget_agent.cli ask "What if I cut TV by 30%?"
python -m budget_agent.cli chat
python -m budget_agent.cli evaluate                               # the 3 evaluation pillars
pytest -q
```

To run the learning loop every Monday at 8:00 (cron):
```
0 8 * * 1 cd /path/to/repo && python -m budget_agent.cli weekly --csv /drops/latest_week.csv
```

## Data format

One row per week, ideally covering about 2 years (104+ weeks):

```
date,sales,spend_facebook,spend_instagram,spend_google_ads,spend_tv,spend_radio,spend_email,holiday,promo
2023-01-02,41234.5,1520.0,980.0,1800.0,0,650.0,210.0,1,0
```

- `spend_<channel>` columns define the channels. Use as many as you like.
- Any other numeric column (`holiday`, `promo`, `price_index`, ...) is treated as an external control, so ads aren't credited for sales they didn't cause.
- `data/demo_company.csv` is a 156-week example. `data/new_weeks_example.csv` holds 8 later weeks for testing the learning loop.
- `data/real_apparel_brand_usd.csv` contains **real data**: 146 weeks (Aug 2021 to May 2024) for an anonymised US apparel brand, with spend on Google Search, Google Performance Max, Facebook, Instagram and other paid channels. Sales = net revenue (list price minus discounts). Controls: holiday weeks and the weekly discount rate.

### Real data source

The real data comes from the public **Conjura "Multi-Region Marketing Mix Modeling (MMM) Dataset for Several eCommerce Brands"** ([Figshare 25314841](https://figshare.com/articles/dataset/Multi-Region_Marketing_Mix_Modeling_MMM_Dataset_for_Several_eCommerce_Brands/25314841)): anonymised daily data for many brands. `scripts/prepare_conjura.py` converts it to our format:

```bash
python scripts/prepare_conjura.py conjura_mmm_data.csv --list
python scripts/prepare_conjura.py conjura_mmm_data.csv --brand <id> -o data/my_brand.csv
```

Results on the real brand: in-sample R² 0.97; 13-week holdout R² **0.90**, MAPE **8.2%**. For a $500K quarter with each channel limited to ±30% of its current budget (the default in the web app), the optimized plan gives **+7.3%** expected sales. Without limits the model suggests much larger moves (+46%, with Google Search and Facebook cut to $0 and 64% of the budget on Instagram; at $100K even +142%). Treat those with caution, because real data is noisier and channels move together. Four other brands in the dataset fitted poorly (negative holdout R²), so they aren't included. I also tested adding the brand's direct, organic-search and email clicks as extra drivers. Every combination made the holdout prediction slightly worse (R² 0.88–0.89 vs 0.90), so they aren't used. Ads partly cause those visits, so they aren't true outside factors.

### Data still missing (needs the partner company)

| Missing | Why it matters |
|---|---|
| Gross margin | Turns revenue into profit (see Profit mode) |
| Offline spend (TV, radio, print, out-of-home) | The real brand is online-only, so TV/radio exist only in the simulated data |
| Prices, competitor activity, stock-outs, product launches | Non-ad drivers. Without them the model gives ads too much credit (about 94% of sales here) |
| Lift tests / experiments | The only way to check each channel's ROI on real data, not just overall accuracy |

## How it works

```
data.py        → load, clean, quality report, simulator with known ground truth
mmm.py         → sales = baseline + trend + seasonality + controls + Σ β·Hill(Adstock(spend))
optimizer.py   → SLSQP multi-start: max incremental sales s.t. budget, min/max, max-change constraints
evaluation.py  → accuracy (holdout R²/MAPE), business value (uplift), validation (truth recovery)
workspace.py   → persistent state + the tools the agent calls
tools.py       → OpenAI tool schemas + dispatcher
agent.py       → LLM loop: plan → tools → act → grounding check → answer
autopilot.py   → same workflows without an LLM (offline / scheduled)
server.py      → FastAPI: JSON API for every tool + streaming chat; serves the React app
frontend/      → React web app (Vite + TypeScript + Tailwind)
app.py         → older Streamlit dashboard
```

**Model.** Geometric adstock (carry-over) and Hill saturation for each channel, plus trend, Fourier seasonality and controls. Differential evolution searches the nonlinear parameters, and bounded least squares solves the linear ones exactly (media effects ≥ 0). Optional recency weighting gives recent weeks more say.

**Saturation point** = the weekly spend beyond which the next $1 returns less than $1 in sales.

## What the number check guarantees

`grounding.py` checks that every figure in the agent's answer was **copied from a tool result**, not invented. It does **not** check whether the agent's reasoning about those numbers is right.

A figure is accepted only if it equals a tool value, allowing for the rounding actually written ("$23.4K" may differ by up to $50, "$23K" by up to $500), a fraction shown as a percentage, a period total shown per week, or a sign change. Sums or differences the agent computes itself are flagged, and the agent then uses a tool value instead. Years, "13 weeks", "Q1" and list numbers are skipped; all other numbers are checked.

How strict it is, measured against realistic tool outputs (600 invented figures): about 2–4% of invented dollar amounts slip through by coincidence. Percentages and ratios written with one decimal slip through more often (about 15%), because tool outputs contain dozens of small values (ROIs, saturation levels) that a random figure can land on. Real figures from tool outputs are all accepted. An earlier version that also accepted sums and differences of value pairs let about 80% of invented figures through.

## Profit mode (gross margin)

An ROI of 2.0 sounds good, but at a 40% gross margin each $1 of ads only brings back $0.80 of profit. Set the margin in the sidebar, in `.env` (`GROSS_MARGIN=0.55`), or tell the agent ("our margin is 55%"). Once it's set:

- break-even becomes 1/margin of sales per ad dollar, and alerts and saturation points use it;
- each channel shows its profit per ad dollar;
- the optimizer can **maximize profit**: the budget becomes a cap, and it stops spending where the next dollar no longer pays for itself.

There's no default margin. A real margin has to come from the company, because public benchmarks vary too much (general retail is around 33%).

In profit mode a "max change" limit only caps *increases*. Cuts are always allowed, so the plan is never forced to keep spending on a channel that loses money.

On the real brand ($500K quarter, increases capped at +30%), the result depends heavily on the margin. At 40% the current split *loses* $11K of profit over a quarter, while the profit plan makes $139K and spends only $211K. At 55%, the current split makes $172K and the profit plan $279K. At 70%, it's $355K vs $431K. Facebook and Google Search don't pay for themselves at any margin from 40% to 70%, so the profit plan cuts them to $0.

The margin you enter in Settings (or give the agent) is saved in the workspace and survives restarts. It takes priority over `GROSS_MARGIN` in `.env`.

## Results on the demo data (reproducible: `python -m budget_agent.cli evaluate`)

| Pillar | Result |
|---|---|
| Accuracy (13-week holdout) | test R² **0.92**, test MAPE **2.1%** |
| Business value ($100K, 13 weeks) | optimized split: **+10.5%** incremental sales vs. the current split |
| Validation (known truth) | ROI rank correlation 0.71, mean ROI error 18%. Judged against the *true* response curves, our plan captures **93%** of the best possible uplift |

The ROI estimates for individual channels are noisy, which is normal for MMM. The validation shows that the *decisions* still come out close to optimal.

## Using Google Meridian / Meta Robyn

The proposal names Meridian and Robyn. This repo ships its own transparent implementation of the same model family, so it runs anywhere in seconds. To use Meridian (Bayesian, `pip install google-meridian`, needs TensorFlow), fit it and copy each channel's posterior-mean `decay`, `half_sat`, `slope` and `beta` into `MMM.params`. The optimizer, agent, alerts and dashboard work unchanged. Robyn is an R package. Its exported hyperparameters can be mapped the same way.
