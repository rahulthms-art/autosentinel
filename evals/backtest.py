"""Historical backtest + tuning of the early-warning detector, with a held-out test set.

Data: NHTSA recalls and owner complaints for 16 vehicles (cached in evals/cache/).
Vehicles are split into TRAIN (used to choose detector settings) and TEST (reported only).

Metrics (per vehicle set):
- hit rate:  share of recalls whose component family was flagged as an unrecalled spike
             at some weekly check in the 120 days before the recall was filed;
- precision: share of alerts (monthly checks, >= 1 year in the past) that were followed by a
             recall on that component family within 12 months. Repeat alerts for the same
             vehicle + family within 180 days count once.
Run: python -m evals.backtest
"""
from __future__ import annotations

import bisect
import itertools
import json
from datetime import date, timedelta
from pathlib import Path

from radar.signals import SpikeParams, component_family, detect_spikes, recall_families

CACHE = Path(__file__).parent / "cache" / "backtest_data.json"
LOOKBACK = timedelta(days=120)


VEHICLES = [("Tesla", "Model Y", range(2020, 2026)), ("Tesla", "Model 3", range(2018, 2026)),
            ("Chevrolet", "Bolt EV", range(2017, 2024)), ("Hyundai", "Ioniq 5", range(2022, 2026)),
            ("Volkswagen", "ID.4", range(2021, 2026)), ("Kia", "EV6", range(2022, 2026)),
            ("Rivian", "R1T", range(2022, 2026)), ("Ford", "F-150", range(2019, 2024)),
            ("Honda", "CR-V", range(2019, 2024)), ("Toyota", "RAV4", range(2019, 2024)),
            ("Jeep", "Grand Cherokee", range(2019, 2024)), ("Chevrolet", "Silverado 1500", range(2019, 2024)),
            ("Honda", "Civic", range(2019, 2024)), ("Hyundai", "Tucson", range(2019, 2024)),
            ("Kia", "Telluride", range(2020, 2024)), ("Subaru", "Outback", range(2019, 2024))]


def fetch():
    from radar.sources import nhtsa
    data = []
    for make, model, years in VEHICLES:
        recs, comps = [], []
        for y in years:
            recs += nhtsa.recalls(make, model, y)
            comps += nhtsa.complaints(make, model, y)
        data.append({"make": make, "model": model, "recalls": recs, "complaints": comps})
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(data))


def load():
    if not CACHE.exists():
        fetch()
    data = json.load(open(CACHE))
    for v in data:
        v["complaints"] = sorted([c for c in v["complaints"] if c.get("filed")], key=lambda c: c["filed"])
        v["_dates"] = [c["filed"] for c in v["complaints"]]
        seen, recs = set(), []
        for r in sorted([r for r in v["recalls"] if r.get("date")], key=lambda r: r["date"]):
            if r["campaign"] not in seen:
                seen.add(r["campaign"])
                recs.append(r)
        v["recalls"] = recs
    train = [v for i, v in enumerate(data) if i % 2 == 0]
    test = [v for i, v in enumerate(data) if i % 2 == 1]
    return train, test


def window(v, as_of: date, p: SpikeParams):
    """Complaints filed in the detector's lookback window (fast slice of the sorted list)."""
    lo = (as_of - timedelta(days=p.recent_days + p.baseline_days)).isoformat()
    return v["complaints"][bisect.bisect_right(v["_dates"], lo):bisect.bisect_right(v["_dates"], as_of.isoformat())]


def evaluate(vehicles, p: SpikeParams, detector=None) -> dict:
    """detector(v, as_of, spike) -> bool is an optional second-stage filter (e.g. Nemotron)."""
    hits = total = flagged = confirmed = 0
    leads, alerts = [], []
    for v in vehicles:
        if not v["complaints"]:
            continue
        first = date.fromisoformat(v["complaints"][0]["filed"])
        recs = v["recalls"]

        def unrecalled(as_of):
            prior = [r for r in recs if r["date"] < as_of.isoformat()]
            return [s for s in detect_spikes(window(v, as_of, p), prior, as_of=as_of, params=p) if not s["recalled"]]

        # hit rate
        for r in recs:
            d = date.fromisoformat(r["date"])
            fams = recall_families(r["component"])
            if not fams or d - first < timedelta(days=365):
                continue
            total += 1
            for back in range(LOOKBACK.days, -1, -7):
                as_of = d - timedelta(days=back)
                s = next((s for s in unrecalled(as_of) if s["component"] in fams), None)
                if s and (detector is None or detector(v, as_of, s)):
                    hits += 1
                    leads.append(back)
                    break
        # precision
        last_alert = {}
        as_of = first + timedelta(days=365)
        while as_of <= date.today() - timedelta(days=365):
            for s in unrecalled(as_of):
                key = s["component"]
                if key in last_alert and (as_of - last_alert[key]).days < 180:
                    continue
                if detector is not None and not detector(v, as_of, s):
                    continue
                last_alert[key] = as_of
                followed = any(key in recall_families(r["component"]) and
                               as_of.isoformat() < r["date"] <= (as_of + timedelta(days=365)).isoformat() for r in recs)
                flagged += 1
                confirmed += followed
                sev = s["severity"]
                alerts.append({"vehicle": f"{v['make']} {v['model']}", "component": key, "as_of": as_of.isoformat(),
                               "ratio": s["ratio"], "n": s["recent"], "p_value": s["p_value"],
                               "safety_events": sev["crash"] + sev["fire"] + sev["injuries"], "confirmed": followed})
            as_of += timedelta(days=30)
    hit_rate = hits / total if total else 0
    precision = confirmed / flagged if flagged else 0
    f1 = 2 * hit_rate * precision / (hit_rate + precision) if hit_rate + precision else 0
    return {"hits": hits, "recalls": total, "hit_rate": hit_rate, "alerts": flagged, "confirmed": confirmed,
            "precision": precision, "f1": f1, "median_lead_days": sorted(leads)[len(leads) // 2] if leads else None,
            "alert_list": alerts}


BASELINE = SpikeParams(recent_days=90, min_recent=3, min_ratio=2.0, max_p=1.0, volume_adjust=False)
GRID = dict(recent_days=[60, 90], min_recent=[3, 5], min_ratio=[1.5, 2.0, 3.0],
            max_p=[1.0, 0.05, 0.01, 0.001], volume_adjust=[False, True])


def fmt(m):
    return (f"hit rate {m['hits']}/{m['recalls']} = {m['hit_rate']:.0%} | precision {m['confirmed']}/{m['alerts']} = "
            f"{m['precision']:.0%} | F1 {m['f1']:.2f} | median lead {m['median_lead_days']}d")


if __name__ == "__main__":
    train, test = load()
    print("TRAIN:", ", ".join(f"{v['make']} {v['model']}" for v in train))
    print("TEST: ", ", ".join(f"{v['make']} {v['model']}" for v in test))
    print("\nOld detector (ratio only), TRAIN:", fmt(evaluate(train, BASELINE)))
    results = []
    for combo in itertools.product(*GRID.values()):
        p = SpikeParams(**dict(zip(GRID, combo)))
        m = evaluate(train, p)
        results.append((m["f1"], p, m))
    results.sort(key=lambda x: -x[0])
    print("\nTop 5 settings on TRAIN:")
    for f1, p, m in results[:5]:
        print(" ", {k: getattr(p, k) for k in GRID}, "->", fmt(m))
    best = results[0][1]
    print("\nHELD-OUT TEST  old:", fmt(evaluate(test, BASELINE)))
    print("HELD-OUT TEST  new:", fmt(evaluate(test, best)))
