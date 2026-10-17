# 📡 AutoSentinel: Recall & Regulation Radar

**An always-on agent platform that monitors vehicle recalls, regulation changes and supplier risk, and detects likely recalls *before* they are announced from spikes in owner complaints.**

Built with **NVIDIA Nemotron** on **Nebius Token Factory**, **Tavily** live search and **NHTSA** official open data.
Hackathon track: **Best Apps and Agents**

---

## The problem

Automotive compliance, quality and aftersales teams must track:
- **recalls** across many regulators (NHTSA, EU Safety Gate, DVSA, KBA...)
- **regulation changes** (UNECE WP.29, EU type-approval, GSR2, Euro 7, R155/R156 cybersecurity, the EU Battery Regulation...)
- **supplier risk** (shortages, quality escapes, insolvencies)

Today this is manual: analysts read regulator sites and newsletters, items get missed, and recall signals show up in owner complaints weeks before anyone acts.

Why not just ask a chatbot? A chatbot only answers when someone asks, knows nothing about your history, can't process thousands of VINs, and can't spot a statistical spike in official complaint data. AutoSentinel does those things.

## What it does

| Capability | How |
|---|---|
| **Watchlists** | Save what you monitor: vehicles and model years, markets, components, suppliers, topics |
| **Scheduled runs** | Each run collects official recalls (NHTSA API) and searches the web (Tavily, limited to official domains per market such as `ec.europa.eu`, `unece.org` and `gov.uk`) |
| **New-only** | Results are de-duplicated against earlier runs, so you see only what is **new** |
| **Triage** | Nemotron (fast tier) scores relevance, type, severity, affected scope and recommended action. Irrelevant items are auto-dismissed. Rule-based severity floors (e.g. NHTSA *park outside*) cannot be lowered by the model |
| **📈 Recall early warning** | Detects **owner-complaint spikes** per component against a 12-month baseline and flags spikes with **no matching recall**. Nemotron (reasoning tier) reads the complaint texts and names the likely failure mode, the risk and the next action |
| **Risk brief** | Nemotron (reasoning tier) writes a cited brief: executive summary, risk table, signals and what to watch next |
| **Findings workflow** | Mark findings new → acknowledged → actioned → dismissed |
| **🚙 Fleet Check** | Upload a CSV of thousands of VINs: batch decode, then match recalls. Flags *do-not-drive* vehicles. Exports a CSV |
| **💬 Ask the Analyst** | A tool-using Nemotron agent (tracked findings, NHTSA recalls and complaints, live web, VIN decoder) that answers with citations |
| **Alerts** | Slack / Teams webhook for new High-severity items and early-warning signals |
| **🧾 Audit trail** | Every run and every model call is logged with model, latency and tokens |

**Example from live data (Oct 2026):** in the starter watchlist, owner complaints about **service brakes** on one popular EV model were running at **6× their baseline over the last 90 days, with no matching recall**, and lane-departure complaints on another were nearly 3×. These are signals to investigate, not conclusions. That's the kind of thing a quality team wants to see first.

## How it uses NVIDIA Nemotron on Nebius Token Factory

```
                 ┌─────────── Nebius Token Factory (OpenAI-compatible) ──────────┐
                 │  ⚡ Nemotron fast tier (Nano/Super)  🧠 Nemotron reasoning tier (Ultra/Super) │
                 └──────────────▲──────────────────────────▲─────────────────────┘
 NHTSA API ─┐                   │ plan queries, triage     │ signal analysis, briefs, analyst agent
 Tavily ────┼─► Collect ─► De-dup ─► Triage ─► Early-warning signals ─► Brief ─► Store ─► Alert
 VIN CSV ───┘                     (SQLite: watchlists, runs, findings, signals, llm_calls)
```

- **Tiered routing:** high-volume structured work (query planning, batched triage) goes to the fast tier. Reasoning-heavy work (failure-mode analysis, briefs, the multi-step agent) goes to the reasoning tier. This keeps runs fast and cheap, and the Runs & Audit tab shows the token split per model.
- **Live model discovery:** the app lists Token Factory's NVIDIA models and picks Nemotron Nano/Ultra automatically. Users can override the choice in the sidebar or with `NEMOTRON_FAST` / `NEMOTRON_DEEP`.
- **Portable agent protocol:** the analyst agent uses a JSON action loop, so it works with any Nemotron model without needing provider-specific function calling.
- **Batch-ready:** `python -m jobs.run_watchlists` is a headless entrypoint for cron, GitHub Actions or **Nebius Serverless Jobs**.

## Design

The UI follows a design system generated from the open-source [UI UX Pro Max](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill) dataset, matched to *Analytics Dashboard* and *Legal/compliance*. The spec is in [`design-system/autosentinel/MASTER.md`](design-system/autosentinel/MASTER.md).
- **Style:** data-dense dashboard with drill-down, built on trust and authority.
- **Colors:** authority navy `#1E3A8A`, amber attention `#B45309`, red critical `#DC2626`.
- **Typography:** IBM Plex Sans and Mono.
- **Icons:** Material icons only, no emoji.
- **Accessibility:** severity is always shown as text, shape and color; focus outlines are visible; reduced motion is respected; every table can be exported as CSV.
- **Early-warning chart:** follows the dataset's anomaly-detection chart rule. It has an alert band for the last 90 days, highlighted spike markers with text labels, a distinct dash style per series, and a table fallback.

## Measured accuracy

`python -m evals.run_evals` scores the tool against ground truth: hand-labelled items, synthetic data with known answers, and NHTSA's API queried directly. AI scenarios are repeated 3 times because model outputs vary. The latest results are in [`evals/RESULTS.md`](evals/RESULTS.md).

| Area | Result (3 runs) |
|---|---|
| Triage: relevance, severity, type (Nemotron Nano) | 115/117 (98%). The only miss is an indirect supplier story scored just below the cut-off |
| Spike detection and recall matching | 26/26 synthetic cases |
| Fleet check: VIN decode and recall counts vs NHTSA | 7/7 |
| Analyst agent: factual answers vs NHTSA (Nemotron Ultra) | 12/12 |
| Analyst agent: refuses to invent non-existent recalls or rules | 9/9 |
| Risk brief: citations that point to real sources | 25/25 |

### Early warning: held-out backtest

`python -m evals.backtest` replays 16 vehicles' NHTSA history (354 recalls, about 25,800 complaints). Settings were chosen on 8 vehicles; results are reported on the **other 8**.

| Held-out vehicles | Recalls flagged up to 120 days early | Alerts followed by a recall within 12 months |
|---|---|---|
| Original detector | 17/100 (17%) | 8% |
| **Optimised detector** | **27/100 (27%)**, median lead 120 days | 11% overall: **Priority 22%**, Watch 8%, Low 4% |

What changed:
1. **Component matching:** "SERVICE BRAKES, HYDRAULIC" and fuel-system recalls never matched complaints, because NHTSA writes commas inside category names. Driver-assist complaints and recalls are now one ADAS family.
2. **A tuned threshold.**
3. **Priority tiers** that rank alerts. On unseen vehicles, Priority alerts preceded a recall about 2× as often as average and 5× as often as Low.

Tested and rejected: a Poisson significance gate, a fleet-growth adjustment, and a Nemotron screen of complaint text. Nemotron judged most spikes "strong", so it did not separate true from false alarms.

**Honest reading:**
- The labelled sets are small and written by the developer.
- A naive rule (alert on any component with 3+ complaints) catches more recalls at similar precision. The app offers it as *High sensitivity*.
- Early warning ranks leads for a quality team to investigate. It is not a recall predictor.

## Cost control

Every Nemotron call is priced and logged. The sidebar shows the estimated spend, and **Runs & audit** shows the cost of each run.
- **Typical cost:** about $0.05 per radar run and about $0.03 per analyst question, using a cautious Ultra price estimate.
- **Hard stop:** `BUDGET_USD` (default $10) blocks AI calls once the estimated spend for the deployment reaches it.
- **Public demo** (`DEMO_MODE=1`):
  - It starts with real stored results (`demo/snapshot.json`), so visitors see the full product without spending anything.
  - It is capped at 3 radar runs and 8 analyst questions per day.
- **Prices:** set real prices from your Token Factory prices page with `PRICE_NANO`, `PRICE_ULTRA` and similar, as `"input,output"` in USD per 1M tokens.

## Run locally

```bash
git clone https://github.com/rahulthms-art/autosentinel.git && cd autosentinel
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # add your keys
streamlit run app.py --server.address localhost   # only reachable from this computer
```

Keys:
- Nebius Token Factory: https://tokenfactory.nebius.com
- Tavily: https://tavily.com

The **Early Warning** charts and **Fleet Check** work without any keys, because NHTSA data is public.

Try Fleet Check with `sample_fleet.csv`.

### Scheduled monitoring
```bash
export NEBIUS_API_KEY=... TAVILY_API_KEY=... ALERT_WEBHOOK_URL=...   # webhook optional
python -m jobs.run_watchlists          # all watchlists
python -m jobs.run_watchlists --id 1   # one watchlist
```
See `.github/workflows/radar-schedule.yml` for a weekday 06:00 schedule.

### Docker
```bash
docker build -t autosentinel .
docker run -p 8501:8501 -v $PWD/data:/data -e NEBIUS_API_KEY=... -e TAVILY_API_KEY=... autosentinel
```

### Tests
```bash
pip install -r requirements-dev.txt && pytest -q
```
The tests cover spike detection, JSON parsing, VIN validation, severity rules, a full end-to-end run with de-duplication and tier routing, failed-run recording, and the analyst tool loop. They use a fake LLM, so they need no keys.

## Project layout

```
ui.py                  Design tokens, severity badges, anomaly chart
app.py                 Streamlit UI (Dashboard, Early Warning, Fleet Check, Analyst, Watchlists, Runs & Audit)
radar/pipeline.py      Run engine: collect → de-dup → triage → signals → brief → store → alert
radar/signals.py       Complaint spike detection
radar/analyst.py       Tool-using Nemotron agent
radar/llm.py           Token Factory client, tiered router, audit hooks
radar/sources/         NHTSA (recalls, complaints, vPIC VIN decode), Tavily, HTTP retries/cache
radar/db.py            SQLite schema & queries
radar/fleet.py         Batch VIN exposure check
radar/alerts.py        Slack/Teams webhook
jobs/run_watchlists.py Headless scheduled runner
tests/                 pytest suite
evals/run_evals.py     Accuracy evaluation (10 scenarios, ground truth)
evals/backtest.py      Early-warning backtest with a train/test split
```

## Limitations and honesty notes

- **Official structured data is US-only (NHTSA).** Europe-wide recall data has no open API (UK DVSA bulk data was last refreshed years ago), so EU/UK coverage uses web search limited to official domains. Adding EU Safety Gate or OEM feeds is the next step.
- Complaint spikes are **signals, not proof**. Low counts are noisy, which is why spikes require at least 3 complaints and at least 2× the baseline.
- Fleet Check matches by make, model and year. Confirm remedy status per VIN on the OEM portal.
- SQLite fits a single team. For multi-team production, swap in Postgres; the DB layer is isolated in `radar/db.py`.
- Decision support only: always confirm with the regulator or OEM source before acting.

## Feedback on Nebius Token Factory / NVIDIA tools

_TODO: fill in after building, for the "Most Valuable Feedback" prize._

## License

MIT. See [LICENSE](LICENSE).
