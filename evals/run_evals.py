"""AutoSentinel accuracy evaluation: 10 scenarios scored against ground truth.

Ground truth comes from (a) hand-labelled items written for this eval, (b) synthetic data with
known answers, and (c) NHTSA's official API queried directly (independently of the app's parser).

Run:  python -m evals.run_evals            (needs NEBIUS_API_KEY + TAVILY_API_KEY, or .streamlit/secrets.toml)
Writes evals/RESULTS.md and evals/results.json.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from datetime import date, timedelta
from pathlib import Path

import requests

from radar.analyst import ask
from radar.config import Settings
from radar.db import DB
from radar.fleet import check_fleet, clean_vins
from radar.llm import Router, list_nvidia_models, make_client, pick_default
from radar.pipeline import triage
from radar.seed import DEFAULT
from radar.signals import detect_spikes, top_level
from radar.sources import nhtsa

ROOT = Path(__file__).resolve().parents[1]
RESULTS: list[dict] = []


def settings_from_env() -> Settings:
    try:
        import toml
        for k, v in toml.load(ROOT / ".streamlit/secrets.toml").items():
            os.environ.setdefault(k, str(v))
    except Exception:
        pass
    return Settings()


def record(n, name, what, correct, total, details, note=""):
    acc = correct / total if total else 0.0
    RESULTS.append({"n": n, "scenario": name, "measures": what, "correct": correct, "total": total,
                    "accuracy": round(acc, 3), "note": note, "details": details})
    print(f"S{n:<2} {name:<42} {correct}/{total} = {acc:.0%}  {note}")


def item(title, summary, source="web"):
    return {"source": source, "title": title, "summary": summary, "severity": None}


# ------------------------------------------------------------------ S1 triage relevance
def s1_relevance(router):
    relevant = [
        item("EU Safety Gate alert: Hyundai Ioniq 5 integrated charging control unit failure",
             "Loss of 12V charging may cause loss of drive power. Vehicles: Hyundai Ioniq 5 2022-2023. Measures: recall."),
        item("Battery passport obligations under EU Battery Regulation apply from February 2027",
             "EV batteries placed on the EU market must carry a digital battery passport including carbon footprint data."),
        item("UNECE WP.29 adopts amendment to UN R155 cybersecurity management systems",
             "Amendment clarifies supplier-chain obligations for vehicle type approval in contracting parties."),
        item("Fire at airbag inflator plant disrupts deliveries to European automakers",
             "Supplier warns of multi-week shortages of airbag inflators used across several EV platforms."),
        item("Volkswagen recalls ID.4 over 12V battery cable chafing against steering column",
             "Cable may short circuit, increasing the risk of a fire. Affects 2023 ID.4 RWD."),
        item("Tesla Model Y recall: front passenger airbag may not deploy correctly",
             "Occupant classification software error may disable the airbag for adult passengers."),
        item("EU General Safety Regulation phase 3 requirements for ADAS from July 2026",
             "New type approvals must include advanced driver distraction warning and emergency lane keeping."),
        item("Chevrolet Bolt EV battery modules: new remedy for thermal events announced",
             "OEM will replace battery modules on 2019-2022 Bolt EV after reports of fires while parked."),
        # harder: relevant but indirect
        item("Euro NCAP 2026 protocols add driver-monitoring and lane-support scoring",
             "New assessment rules for driver monitoring cameras and lane keeping systems from 2026."),
        item("Lithium price slump squeezes margins at major battery cell makers",
             "Two leading cell suppliers warn of plant closures and delayed EV battery deliveries."),
    ]
    irrelevant = [
        item("Recall of magnetic building toys due to swallowing hazard", "Children's toy sold in toy stores."),
        item("Harley-Davidson recalls touring motorcycles over front fork bolts", "Motorcycles 2021 model year."),
        item("Peanut butter recalled for possible salmonella contamination", "Food product sold in supermarkets."),
        item("Football club completes record striker transfer", "Sports news."),
        item("Boeing 737 MAX door plug inspections ordered by FAA", "Aviation safety directive."),
        item("Tablet battery swelling: manufacturer offers free replacements", "Consumer electronics recall."),
        item("Celebrity chef opens new restaurant in Paris", "Lifestyle news."),
        item("Dishwasher recall over door latch that can break", "Household appliance recall."),
        # harder: look-alikes that are out of scope
        item("Electric scooter battery fires prompt recall of 40,000 e-scooters",
             "Lithium-ion packs in a consumer e-scooter brand may overheat; not road vehicles."),
        item("Diesel pickup recall: fuel injection pump may fail on V8 engines",
             "Affects diesel pickup trucks only; no electric models included."),
    ]
    items = relevant + irrelevant
    triage(router, DEFAULT, items, lambda m: None)
    details, correct = [], 0
    for i, it in enumerate(items):
        expected = i < len(relevant)
        predicted = it["score"] >= 4
        correct += expected == predicted
        details.append({"title": it["title"], "expected_relevant": expected, "score": it["score"], "ok": expected == predicted})
    record(1, "Triage: relevance filtering", "Relevant vs irrelevant (score >= 4)", correct, len(items), details)


# ------------------------------------------------------------------ S2 severity
def s2_severity(router):
    high = [
        ("Battery may overheat and catch fire; owners told to park outside", {"park_outside": True, "component": "ELECTRICAL SYSTEM", "consequence": "fire"}),
        ("Driver frontal airbag may not deploy in a crash", {"component": "AIR BAGS:FRONTAL", "consequence": "increased injury"}),
        ("Brake caliper bolts may loosen causing loss of braking", {"component": "SERVICE BRAKES", "consequence": "crash"}),
        ("Steering column coupling may separate causing loss of steering", {"component": "STEERING", "consequence": "crash"}),
        ("Inverter fault may cause sudden loss of drive power at highway speed", {"component": "ELECTRICAL SYSTEM:PROPULSION", "consequence": "loss of drive power increasing the risk of a crash"}),
    ]
    not_high = [
        ("Owner's manual is missing child seat installation information", {"component": "EQUIPMENT", "consequence": "improper installation"}),
        ("Tire placard label lists incorrect tire size", {"component": "LABELS", "consequence": "improper inflation"}),
        ("Floor mat clip may be missing from the passenger side", {"component": "EQUIPMENT", "consequence": "mat may slide"}),
        ("Seat belt reminder chime may not sound for rear passengers", {"component": "SEAT BELTS:WARNING", "consequence": "noncompliance with FMVSS 208 chime"}),
        ("Cargo cover may rattle at highway speed", {"component": "EQUIPMENT", "consequence": "noise"}),
    ]
    items = []
    for text, raw in high + not_high:
        it = item(f"Recall: {text}", text, source="nhtsa")
        it["severity"] = nhtsa.recall_severity(raw)  # the app's rule-based floor, as in the real pipeline
        items.append(it)
    triage(router, DEFAULT, items, lambda m: None)
    details, correct = [], 0
    for i, it in enumerate(items):
        expected_high = i < len(high)
        ok = (it["severity"] == "High") == expected_high
        correct += ok
        details.append({"title": it["title"], "expected": "High" if expected_high else "Medium/Low",
                        "got": it["severity"], "ok": ok})
    record(2, "Triage: severity rating", "Safety-critical = High; minor = Medium/Low", correct, len(items), details)


# ------------------------------------------------------------------ S3 type classification
def s3_type(router):
    labelled = [
        ("recall", "Transport Canada recall 2026-445: Hyundai Ioniq 5 seat belt pretensioner", "Owners will be notified; dealers will replace the pretensioner."),
        ("recall", "EU Safety Gate: Volkswagen ID.4 high-voltage battery recall", "Risk of fire. Measures taken: recall of the product from end users."),
        ("recall", "DVSA recall: Tesla Model Y rear camera image may not display", "Vehicle recall notice R/2026/101."),
        ("regulation", "Commission Delegated Regulation (EU) 2026/1753 on battery durability requirements", "Lays down minimum durability performance for EV batteries."),
        ("regulation", "UN Regulation No. 156 amendment on software update management", "Applies to new vehicle types from 2027 in contracting parties."),
        ("regulation", "UK Vehicle Emissions Trading Schemes: 2027 ZEV mandate targets confirmed", "Manufacturers must reach 33% zero-emission new car sales."),
        ("supplier", "Battery cell maker files for insolvency, threatening EV production in Europe", "Customers include several European carmakers."),
        ("supplier", "Chip shortage: semiconductor supplier cuts automotive MCU allocations by 20%", "Automakers warn of production slowdowns."),
        ("supplier", "Flood halts production at Thai plant making wiring harnesses for EVs", "Supplier expects two-week outage."),
    ]
    items = [item(t, s) for _, t, s in labelled]
    triage(router, DEFAULT, items, lambda m: None)
    details, correct = [], 0
    for (exp, t, _), it in zip(labelled, items):
        ok = it["type"] == exp
        correct += ok
        details.append({"title": t, "expected": exp, "got": it["type"], "ok": ok})
    record(3, "Triage: type classification", "recall / regulation / supplier", correct, len(items), details)


# ------------------------------------------------------------------ S4 spike detector (synthetic)
def s4_spikes():
    rng = random.Random(42)
    as_of = date(2026, 10, 1)
    cases = []
    for i in range(20):
        comp = f"COMPONENT {i}"
        spike = i < 10
        baseline_per_90d = rng.choice([1, 2, 3, 4])
        comps = []
        for d in range(90, 455):  # 12-month baseline window
            if rng.random() < baseline_per_90d / 90:
                comps.append(as_of - timedelta(days=d))
        recent_rate = baseline_per_90d * (rng.uniform(3, 6) if spike else rng.uniform(0.5, 1.3))
        recent_n = max(3 if spike else 0, round(recent_rate))
        comps += [as_of - timedelta(days=rng.randint(1, 89)) for _ in range(recent_n)]
        cases.append((comp, spike, [{"filed": d.isoformat(), "components": [comp], "summary": "", "crash": False,
                                     "fire": False, "injuries": 0, "deaths": 0} for d in comps]))
    details, correct = [], 0
    tp = fp = fn = 0
    for comp, spike, comps in cases:
        flagged = any(s["component"] == comp for s in detect_spikes(comps, [], as_of=as_of))
        ok = flagged == spike
        correct += ok
        tp += flagged and spike
        fp += flagged and not spike
        fn += spike and not flagged
        details.append({"component": comp, "true_spike": spike, "flagged": flagged, "ok": ok})
    note = f"precision {tp / max(tp + fp, 1):.0%}, recall {tp / max(tp + fn, 1):.0%}"
    record(4, "Early warning: spike detection (synthetic)", "Injected 3-6x spikes vs normal noise", correct, len(cases), details, note)


# ------------------------------------------------------------------ S5 recall-coverage matching
def s5_recall_matching():
    as_of = date(2026, 10, 1)
    comps = [{"filed": f"2026-09-{d:02d}", "components": ["AIR BAGS:FRONTAL"], "summary": "", "crash": False,
              "fire": False, "injuries": 0, "deaths": 0} for d in range(1, 9)]
    cases = [
        ("Recall on same component 2 months ago", [{"component": "AIR BAGS:SIDE", "date": "2026-08-01"}], True),
        ("Recall on same component 11 months ago", [{"component": "AIR BAGS", "date": "2025-11-15"}], True),
        ("Recall on same component 3 years ago (stale)", [{"component": "AIR BAGS", "date": "2023-06-01"}], False),
        ("Recall on a different component", [{"component": "SERVICE BRAKES", "date": "2026-08-01"}], False),
        ("No recalls at all", [], False),
        ("Two recalls, one matching", [{"component": "STEERING", "date": "2026-07-01"},
                                       {"component": "AIR BAGS:CURTAIN", "date": "2026-02-01"}], True),
    ]
    details, correct = [], 0
    for name, recalls, expected in cases:
        spikes = detect_spikes(comps, recalls, as_of=as_of)
        got = bool(spikes and spikes[0]["recalled"])
        correct += got == expected
        details.append({"case": name, "expected_covered": expected, "got": got, "ok": got == expected})
    record(5, "Early warning: recall-coverage matching", "Is the spiking component already recalled?", correct, len(cases), details)


# ------------------------------------------------------------------ S6 fleet check
def _nhtsa_direct_count(make, model, year):
    r = requests.get("https://api.nhtsa.gov/recalls/recallsByVehicle",
                     params={"make": make, "model": model, "modelYear": year}, timeout=30)
    return len({x["NHTSACampaignNumber"] for x in r.json().get("results", [])})


def s6_fleet():
    expected = {"5YJYGDEE3MF100000": ("Tesla", "Model Y", 2021),
                "1G1FY6S02M4100000": ("Chevrolet", "Bolt EV", 2021),
                "KM8KN4AE6NU100000": ("Hyundai", "Ioniq 5", 2022)}
    invalid = ["5YJYGDEE3MF10000", "IOQ00000000000000", "HELLO-WORLD-VIN!!"]
    good, bad = clean_vins(list(expected) + invalid)
    details, correct, total = [], 0, 0
    total += 1
    ok = sorted(bad) == sorted(invalid) and sorted(good) == sorted(expected)
    correct += ok
    details.append({"check": "invalid VINs rejected, valid kept", "ok": ok})
    for row in check_fleet(good):
        exp = expected[row["VIN"]]
        total += 2
        ok_decode = (row["Make"], row["Model"], row["Year"]) == exp
        truth = _nhtsa_direct_count(*exp)
        ok_count = row["Recalls"] == truth
        correct += ok_decode + ok_count
        details.append({"vin": row["VIN"], "decoded": [row["Make"], row["Model"], row["Year"]], "expected": exp,
                        "decode_ok": ok_decode, "recalls": row["Recalls"], "nhtsa_direct": truth, "count_ok": ok_count})
    record(6, "Fleet check: VIN decode + recall match", "Decode, invalid-VIN rejection, recall counts vs NHTSA", correct, total, details)


# ------------------------------------------------------------------ S7 + S8 analyst agent
def _analyst(db, router, settings, wl_id, q):
    steps = []
    t0 = time.time()
    ans = ask(db, router, settings, wl_id, q, [], on_step=steps.append)
    return ans, [s.get("tool") for s in steps if s.get("tool")], round(time.time() - t0, 1)


def s7_analyst_facts(db, router, settings, wl_id):
    bolt_n = _nhtsa_direct_count("Chevrolet", "Bolt EV", 2022)
    id4 = requests.get("https://api.nhtsa.gov/recalls/recallsByVehicle",
                       params={"make": "Volkswagen", "model": "ID.4", "modelYear": 2023}, timeout=30).json()["results"]
    id4_campaigns = {x["NHTSACampaignNumber"] for x in id4}
    yl = requests.get("https://api.nhtsa.gov/recalls/recallsByVehicle",
                      params={"make": "Tesla", "model": "Model Y", "modelYear": 2022}, timeout=30).json()["results"]
    airbag = sorted({x["NHTSACampaignNumber"] for x in yl if "AIR BAG" in x["Component"].upper()})
    questions = [
        (f"How many NHTSA recall campaigns exist for the 2022 Chevrolet Bolt EV? Reply with the number.",
         lambda a: str(bolt_n) in re.findall(r"\b\d+\b", a), f"{bolt_n} campaigns"),
        ("Give one NHTSA campaign number for a recall of the 2023 Volkswagen ID.4.",
         lambda a: any(c in a for c in id4_campaigns), f"any of {len(id4_campaigns)} real campaigns"),
        ("Decode VIN 5YJYGDEE3MF100000: what make, model and model year is it?",
         lambda a: "tesla" in a.lower() and "model y" in a.lower() and "2021" in a, "Tesla Model Y 2021"),
        ("Does the 2022 Tesla Model Y have any NHTSA recall involving air bags? Give the campaign number(s) or say none.",
         (lambda a: any(c in a for c in airbag)) if airbag else (lambda a: bool(re.search(r"\bno\b|none|not", a.lower()))),
         ", ".join(airbag) or "none"),
    ]
    details, correct = [], 0
    for q, check, truth in questions:
        ans, tools, secs = _analyst(db, router, settings, wl_id, q)
        ok = bool(check(ans))
        correct += ok
        details.append({"question": q, "ground_truth": truth, "tools_used": tools, "seconds": secs,
                        "answer": ans[:600], "ok": ok})
    record(7, "Analyst agent: factual questions", "Answers match NHTSA ground truth", correct, len(questions), details)


HONEST = re.compile(r"no (recall|record|data|information|evidence|result|match)|not (find|found|exist|available|able)|"
                    r"could ?n[o']t|cannot|unable|does not exist|doesn't exist|no such|non-?existent|"
                    r"isn't|is not (a |an )?(valid|real|listed)|did not return|returned no|nothing|empty|\b0 recalls|zero recalls|"
                    r"\bno\b[^.]{0,80}\b(was|were|is|are|could be|has been)\s+(found|located|identified)", re.I)


def s8_analyst_hallucination(db, router, settings, wl_id):
    questions = [
        "Summarise NHTSA recall campaign 26V999000 for the 2024 Acme Zephyr.",
        "How many recalls does the 2031 Chevrolet Bolt EV have?",
        "What fine did the EU Commission impose on Tesla under Regulation (EU) 2099/999?",
    ]
    honest = HONEST
    details, correct = [], 0
    for q in questions:
        ans, tools, secs = _analyst(db, router, settings, wl_id, q)
        ok = bool(honest.search(ans))
        correct += ok
        details.append({"question": q, "tools_used": tools, "seconds": secs, "answer": ans[:600], "ok": ok})
    record(8, "Analyst agent: refuses to invent facts", "Says 'not found' for non-existent recalls/rules",
           correct, len(questions), details)


# ------------------------------------------------------------------ S9 brief citation faithfulness
def s9_citations(settings):
    db = DB(settings.db_path)
    wls = [w for w in db.watchlists() if db.runs(w["id"])]
    details, correct, total = [], 0, 0
    for wl in wls:
        run = next((r for r in db.runs(wl["id"]) if r["status"] == "ok" and r["brief"] and "No new items" not in r["brief"]), None)
        if not run:
            continue
        brief = run["brief"]
        ids = {f["id"] for f in db.findings(wl["id"])}
        texts = " ".join((f["title"] or "") + " " + (f["summary"] or "") + " " + json.dumps(f.get("raw") or "")
                         for f in db.findings(wl["id"]))
        n_signals = len([s for s in db.q("SELECT id FROM signals WHERE run_id=?", (run["id"],))])
        f_refs = {int(x) for x in re.findall(r"\[F(\d+)\]", brief)} | \
                 {int(x) for grp in re.findall(r"\[F(\d+(?:\]\s*\[F\d+)*)", brief) for x in re.findall(r"\d+", grp)}
        s_refs = {int(x) for x in re.findall(r"\[S(\d+)\]", brief)}
        campaigns = set(re.findall(r"\b\d{2}V\d{6}\b", brief))
        for ref in sorted(f_refs):
            total += 1
            correct += ref in ids
        for ref in sorted(s_refs):
            total += 1
            correct += 1 <= ref <= max(n_signals, 0)
        for c in sorted(campaigns):
            total += 1
            correct += c in texts
        details.append({"watchlist_id": wl["id"], "run": run["id"], "finding_refs": len(f_refs),
                        "invalid_finding_refs": sorted(r for r in f_refs if r not in ids),
                        "signal_refs": sorted(s_refs), "signals_in_run": n_signals,
                        "campaign_numbers": len(campaigns),
                        "unsupported_campaign_numbers": sorted(c for c in campaigns if c not in texts)})
    record(9, "Risk brief: citation faithfulness", "Every [F#]/[S#]/campaign no. exists in the sources",
           correct, total, details, "on the latest real runs in the database")


# ------------------------------------------------------------------ S10 historical backtest
def s10_backtest():
    """Held-out backtest: detector settings were chosen on 8 TRAIN vehicles; these are the 8 TEST vehicles."""
    from evals.backtest import evaluate, load
    from radar.signals import DEFAULT_PARAMS, priority
    _, test = load()
    m = evaluate(test, DEFAULT_PARAMS)
    tiers = {}
    for a in m["alert_list"]:
        t = priority({"p_value": a["p_value"], "severity": {"crash": a["safety_events"], "fire": 0, "injuries": 0}})
        c, n = tiers.get(t, (0, 0))
        tiers[t] = (c + a["confirmed"], n + 1)
    tier_txt = ", ".join(f"{t} {c}/{n} = {c / n:.0%}" for t, (c, n) in sorted(tiers.items(), key=lambda x: -x[1][0] / x[1][1]))
    note = (f"8 held-out vehicles. Alert precision {m['confirmed']}/{m['alerts']} = {m['precision']:.0%}; "
            f"by tier: {tier_txt}; median lead {m['median_lead_days']} days. Most recalls come from OEM testing, "
            "not owner complaints, so a high hit rate is not expected")
    record(10, "Early warning: held-out historical backtest", "Past recalls preceded by a flagged spike (120 days)",
           m["hits"], m["recalls"], {k: v for k, v in m.items() if k != "alert_list"}, note)


# ------------------------------------------------------------------ main
def main():
    settings = settings_from_env()
    if settings.missing():
        raise SystemExit(f"Missing keys: {settings.missing()}")
    client = make_client(settings.nebius_api_key, settings.nebius_base_url)
    models = list_nvidia_models(client)
    fast = settings.fast_model or pick_default(models, ["nano", "super", "nemotron"])
    deep = settings.deep_model or pick_default(models, ["ultra", "super", "nemotron"])
    router = Router(client, fast, deep)
    print(f"Models: fast={fast}  deep={deep}\n")

    mem = DB(":memory:")
    wl_id = mem.save_watchlist("eval", DEFAULT)

    repeats = int(os.getenv("EVAL_REPEATS", "3"))
    t0 = time.time()
    ai = [lambda: s1_relevance(router), lambda: s2_severity(router), lambda: s3_type(router),
          lambda: s7_analyst_facts(mem, router, settings, wl_id), lambda: s8_analyst_hallucination(mem, router, settings, wl_id)]
    for r in range(repeats):  # model outputs vary between runs, so AI scenarios are repeated
        print(f"--- AI scenarios, run {r + 1}/{repeats}")
        for fn in ai:
            fn()
    print("--- deterministic scenarios")
    s4_spikes()
    s5_recall_matching()
    s6_fleet()
    s9_citations(settings)
    s10_backtest()
    elapsed = round(time.time() - t0)
    merged = {}
    for r in RESULTS:
        m = merged.setdefault(r["n"], {**r, "correct": 0, "total": 0, "runs": [], "details": []})
        m["correct"] += r["correct"]
        m["total"] += r["total"]
        m["runs"].append(f"{r['correct']}/{r['total']}")
        m["details"].append(r["details"])
    RESULTS[:] = []
    for n in sorted(merged):
        m = merged[n]
        m["accuracy"] = round(m["correct"] / m["total"], 3) if m["total"] else 0
        if len(m["runs"]) > 1:
            m["note"] = (f"{len(m['runs'])} runs: " + ", ".join(m["runs"]) + ("; " + m["note"] if m["note"] else ""))
        RESULTS.append(m)

    tokens = sum((c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0) for c in router.trace)
    out = {"date": date.today().isoformat(), "fast_model": fast, "deep_model": deep, "seconds": elapsed,
           "llm_calls": len(router.trace), "tokens": tokens, "scenarios": RESULTS}
    (ROOT / "evals/results.json").write_text(json.dumps(out, indent=2, default=str))

    lines = [f"# AutoSentinel accuracy evaluation\n",
             f"Run {out['date']} · fast model `{fast}` · reasoning model `{deep}` · "
             f"{len(router.trace)} model calls · {tokens:,} tokens · {elapsed}s · AI scenarios repeated {repeats}x\n",
             "| # | Scenario | What is measured | Result | Accuracy | Note |", "|---|---|---|---|---|---|"]
    for r in RESULTS:
        lines.append(f"| {r['n']} | {r['scenario']} | {r['measures']} | {r['correct']}/{r['total']} | "
                     f"**{r['accuracy']:.0%}** | {r['note']} |")
    lines.append("\nFull per-case details: `evals/results.json`. Re-run: `python -m evals.run_evals`.")
    (ROOT / "evals/RESULTS.md").write_text("\n".join(lines) + "\n")
    print(f"\nDone in {elapsed}s · {len(router.trace)} model calls · {tokens:,} tokens")


if __name__ == "__main__":
    main()
