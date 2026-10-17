"""Recall early-warning: detect abnormal increases in owner complaints per component.

A component is flagged when complaints filed in the recent window are well above what its
trailing baseline predicts. Spikes on components with no existing recall are the interesting
ones: they are candidate *future* recalls.

Detector (tuned on a held-out-validated backtest, see evals/backtest.py):
- components are normalised to NHTSA component *families* so complaints and recalls match
  ("SERVICE BRAKES, HYDRAULIC:CALIPER" and "SERVICE BRAKES" are the same family);
- driver-assistance complaint categories map to one ADAS family, matching recalls filed under
  "ELECTRICAL SYSTEM:ADAS:...";
- every spike gets a Poisson p-value and a priority tier: on the backtest, very significant spikes
  or spikes with crash/fire/injury reports became recalls about twice as often as the rest.
  (A Poisson *gate* and fleet-growth adjustment were tested and did not help, so they are off.)
"""
from __future__ import annotations

import math
from functools import lru_cache
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta

# Fragments produced when NHTSA's comma-separated component lists split names that themselves
# contain commas ("SERVICE BRAKES, HYDRAULIC"), plus non-component labels.
_DROP = {"HYDRAULIC", "GASOLINE", "DIESEL", "AIR", "ELECTRIC", "FIRERELATED", "SHELL", "BASE",
         "CARRY HANDLE", "CHEST CLIP", "BUCKLE", "HARNESS", "INSERT", "PADDING", "TETHER", "OTHER",
         "UNKNOWN OR OTHER", "OTHER/I AM NOT SURE", "ROLLOVER", "CHILD SEAT",
         "LOWER ANCHOR (ON CAR SEAT OR VEHICLE)"}
_FAMILIES = [  # (prefix, family): first match wins
    ("SERVICE BRAKES", "SERVICE BRAKES"), ("PARKING BRAKE", "PARKING BRAKE"),
    ("FUEL/PROPULSION", "FUEL/PROPULSION SYSTEM"), ("FUEL SYSTEM", "FUEL/PROPULSION SYSTEM"),
    ("HYBRID PROPULSION", "FUEL/PROPULSION SYSTEM"), ("ENGINE", "ENGINE"),
    ("VISIBILITY", "VISIBILITY"), ("ELECTRONIC STABILITY", "STABILITY/TRACTION CONTROL"),
    ("TRACTION CONTROL", "STABILITY/TRACTION CONTROL"),
    # Driver-assistance: owners report these categories, while NHTSA usually files the matching
    # recalls under "ELECTRICAL SYSTEM:ADAS:..." (see recall_families)
    ("FORWARD COLLISION AVOIDANCE", "ADAS"), ("LANE DEPARTURE", "ADAS"),
    ("VEHICLE SPEED CONTROL", "ADAS"), ("ADAPTIVE CRUISE", "ADAS"),
]


@lru_cache(maxsize=4096)
def component_family(component: str) -> str | None:
    """Canonical NHTSA component family, or None for fragments/unknown labels."""
    head = component.split(":")[0].strip().upper()
    if not head or head in _DROP or head.startswith("HYDRAULIC"):
        return None
    for prefix, family in _FAMILIES:
        if head.startswith(prefix):
            return family
    return head


top_level = component_family  # backwards-compatible name


@lru_cache(maxsize=4096)
def recall_families(component: str) -> frozenset:
    """Families a recall covers: its head family, plus ADAS for driver-assistance recalls filed
    under another head (e.g. "ELECTRICAL SYSTEM:ADAS:AUTONOMOUS/SELF DRIVING:SOFTWARE")."""
    fams = {component_family(component)}
    upper = component.upper()
    if ":ADAS" in upper or "LANE" in upper or "COLLISION AVOIDANCE" in upper or "CRUISE" in upper:
        fams.add("ADAS")
    return frozenset(f for f in fams if f)


def complaint_families(c: dict) -> set[str]:
    return {f for f in (component_family(x) for x in c.get("components", [])) if f}


def recall_data_suspect(complaints: list[dict], recalls: list[dict], min_complaints: int = 50) -> bool:
    """Many complaints but zero recalls usually means the recall lookup missed (e.g. NHTSA fails on
    model names with hyphens), so "no recall" claims would be false alarms."""
    return not recalls and len(complaints) >= min_complaints


def monthly_series(complaints: list[dict]) -> dict[str, Counter]:
    """{component family: Counter({'2026-07': 4, ...})}"""
    series: dict[str, Counter] = defaultdict(Counter)
    for c in complaints:
        if not c.get("filed"):
            continue
        for fam in complaint_families(c):
            series[fam][c["filed"][:7]] += 1
    return series


def poisson_sf(n: int, lam: float) -> float:
    """P(X >= n) for X ~ Poisson(lam)."""
    if n <= 0:
        return 1.0
    term, cdf = math.exp(-lam), 0.0
    for k in range(n):
        cdf += term
        term *= lam / (k + 1)
    return max(0.0, 1.0 - cdf)


@dataclass(frozen=True)
class SpikeParams:
    recent_days: int = 90
    baseline_days: int = 365
    min_recent: int = 3          # never alert on fewer complaints than this
    min_ratio: float = 1.5       # observed / expected
    max_p: float = 1.0           # Poisson gate (1.0 = off: it did not improve the backtest)
    volume_adjust: bool = False  # scale baseline by fleet growth (off: did not improve the backtest)
    min_expected: float = 0.5    # floor on expected count (avoids dividing by ~0)


DEFAULT_PARAMS = SpikeParams()


def priority(spike: dict) -> str:
    """Priority tier from the backtest: 'Priority' alerts were confirmed by a recall ~2x as often."""
    sev = spike["severity"]
    events = sev["crash"] + sev["fire"] + sev["injuries"]
    if spike["p_value"] < 0.001 or events >= 3:
        return "Priority"
    if spike["p_value"] < 0.05 or events >= 1:
        return "Watch"
    return "Low"


def detect_spikes(complaints: list[dict], recalls: list[dict], as_of: date | None = None,
                  params: SpikeParams = DEFAULT_PARAMS, **overrides) -> list[dict]:
    p = SpikeParams(**{**params.__dict__, **overrides}) if overrides else params
    as_of = as_of or date.today()
    recent_start = as_of - timedelta(days=p.recent_days)
    base_start = recent_start - timedelta(days=p.baseline_days)
    windows = p.baseline_days / p.recent_days

    recent, base = Counter(), Counter()
    recent_total = base_total = 0
    severity = defaultdict(lambda: {"crash": 0, "fire": 0, "injuries": 0, "deaths": 0})
    examples = defaultdict(list)
    for c in complaints:
        if not c.get("filed"):
            continue
        filed = date.fromisoformat(c["filed"])
        fams = complaint_families(c)
        if recent_start < filed <= as_of:
            recent_total += 1
            for fam in fams:
                recent[fam] += 1
                sev = severity[fam]
                sev["crash"] += c.get("crash", False)
                sev["fire"] += c.get("fire", False)
                sev["injuries"] += c.get("injuries", 0)
                sev["deaths"] += c.get("deaths", 0)
                if len(examples[fam]) < 12:
                    examples[fam].append((c.get("summary") or "")[:600])
        elif base_start < filed <= recent_start:
            base_total += 1
            for fam in fams:
                base[fam] += 1

    # Overall complaint growth (e.g. more vehicles on the road); capped so one noisy quarter
    # cannot hide everything.
    growth = 1.0
    if p.volume_adjust and base_total:
        growth = min(max((recent_total) / (base_total / windows), 0.5), 4.0)

    recalled = {f for r in recalls if r.get("date") and date.fromisoformat(r["date"]) >= base_start
                for f in recall_families(r["component"])}

    spikes = []
    for fam, n in recent.items():
        baseline = base[fam] / windows
        expected = max(baseline * growth, p.min_expected)
        ratio = n / expected
        pval = poisson_sf(n, expected)
        if n >= p.min_recent and ratio >= p.min_ratio and pval <= p.max_p:
            spikes.append({
                "component": fam, "recent": n, "baseline": round(baseline, 2), "expected": round(expected, 2),
                "ratio": round(ratio, 1), "p_value": pval, "recalled": fam in recalled,
                "severity": dict(severity[fam]), "examples": examples[fam],
            })
            spikes[-1]["priority"] = priority(spikes[-1])
    # Unrecalled + safety-relevant + strongest evidence first
    rank = {"Priority": 0, "Watch": 1, "Low": 2}
    spikes.sort(key=lambda s: (s["recalled"], rank[s["priority"]], s["p_value"]))
    return spikes
