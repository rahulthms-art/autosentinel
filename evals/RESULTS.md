# AutoSentinel accuracy evaluation

Run 2026-10-17 · fast model `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` · reasoning model `nvidia/Nemotron-3-Ultra-550b-a55b` · 84 model calls · 263,709 tokens · 554s · AI scenarios repeated 3x

| # | Scenario | What is measured | Result | Accuracy | Note |
|---|---|---|---|---|---|
| 1 | Triage: relevance filtering | Relevant vs irrelevant (score >= 4) | 58/60 | **97%** | 3 runs: 19/20, 20/20, 19/20 |
| 2 | Triage: severity rating | Safety-critical = High; minor = Medium/Low | 30/30 | **100%** | 3 runs: 10/10, 10/10, 10/10 |
| 3 | Triage: type classification | recall / regulation / supplier | 27/27 | **100%** | 3 runs: 9/9, 9/9, 9/9 |
| 4 | Early warning: spike detection (synthetic) | Injected 3-6x spikes vs normal noise | 20/20 | **100%** | precision 100%, recall 100% |
| 5 | Early warning: recall-coverage matching | Is the spiking component already recalled? | 6/6 | **100%** |  |
| 6 | Fleet check: VIN decode + recall match | Decode, invalid-VIN rejection, recall counts vs NHTSA | 7/7 | **100%** |  |
| 7 | Analyst agent: factual questions | Answers match NHTSA ground truth | 12/12 | **100%** | 3 runs: 4/4, 4/4, 4/4 |
| 8 | Analyst agent: refuses to invent facts | Says 'not found' for non-existent recalls/rules | 9/9 | **100%** | 3 runs: 3/3, 3/3, 3/3 |
| 9 | Risk brief: citation faithfulness | Every [F#]/[S#]/campaign no. exists in the sources | 25/25 | **100%** | on the latest real runs in the database |
| 10 | Early warning: held-out historical backtest | Past recalls preceded by a flagged spike (120 days) | 27/100 | **27%** | 8 held-out vehicles. Alert precision 35/333 = 11%; by tier: Priority 18/80 = 22%, Watch 13/154 = 8%, Low 4/99 = 4%; median lead 120 days. Most recalls come from OEM testing, not owner complaints, so a high hit rate is not expected |

Full per-case details: `evals/results.json`. Re-run: `python -m evals.run_evals`.
