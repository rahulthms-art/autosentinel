"""RDW (Netherlands vehicle authority) open data: official recall register, recall status per
licence plate, and the national vehicle register. No API key needed; public domain (CC0).

The RDW is also a type-approval authority for many manufacturers selling across the EU, so its
recall register often reflects EU-wide campaigns. Texts are in Dutch; Nemotron triage writes the
English summary.
"""
from __future__ import annotations

import re
from datetime import date

from .http import get_json

BASE = "https://opendata.rdw.nl/resource/"
RECALLS = BASE + "j9yg-7rg9.json"        # Terugroep_actie: one row per recall
RISK = BASE + "9ihi-jgpf.json"           # Terugroep_actie_risico: danger code per recall
MAKE_TYPE = BASE + "mu2x-mu5e.json"      # Terugroep_voertuig_merk_type: make/type per recall
STATUS = BASE + "t49b-isb7.json"         # Terugroep_actie_status: per plate, open (O) or repaired (P)
REGISTER = BASE + "m9d7-ebf2.json"       # Gekentekende_voertuigen: national vehicle register

DANGER = {"ONG": "accident with injury", "BRA": "fire with injury", "TEL": "higher injury risk in a crash",
          "MIL": "environmental harm", "NNB": "not yet known"}
RISK_LEVEL = {"ERN": "serious", "GEM": "moderate"}
CATEGORY_EN = [  # Dutch defect categories -> English (first match wins)
    ("elektrische", "electrical system"), ("rem", "brakes"), ("stuur", "steering"), ("motor", "engine"),
    ("aandrijving", "powertrain"), ("brandstof", "fuel system"), ("licht", "lighting"), ("wiel", "wheels"),
    ("band", "tyres"), ("gordel", "seat belts"), ("airbag", "airbags"), ("zitplaats", "seats"),
    ("carrosserie", "body"), ("ophanging", "suspension"), ("vering", "suspension"), ("ruit", "glazing"),
    ("spiegel", "mirrors"), ("koppeling", "coupling"), ("uitlaat", "exhaust"), ("software", "software"),
    ("verwarming", "heating"), ("deur", "doors")]


def category_en(dutch: str) -> str:
    d = (dutch or "").lower()
    return next((en for nl, en in CATEGORY_EN if nl in d), d or "other")
PLATE_RE = re.compile(r"^[A-Z0-9]{6}$")


def _q(url: str, **params) -> list[dict]:
    return get_json(url, {("$" + k): v for k, v in params.items()})


def _norm(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def model_matches(rdw_type: str, model: str) -> bool:
    """'ID4' matches 'ID.4'; a family name such as 'IONIQ' matches 'Ioniq 5' (newer variants are
    often registered under the family name); 'IONIQ 6' does not match 'Ioniq 5'."""
    t, m = _norm(rdw_type), _norm(model)
    if not t or not m:
        return False
    return t == m or (len(t) >= 4 and m.startswith(t) and not t[-1].isdigit())


def _quote(values) -> str:
    return ",".join("'" + str(v).replace("'", "''") + "'" for v in values)


def recalls(make: str, model: str, since: date | None = None) -> list[dict]:
    """Official RDW recalls for a make/model, newest first (optionally only those published since a date)."""
    types = _q(MAKE_TYPE, select="referentiecode_rdw,type", where=f"upper(merk)='{make.upper()}'", limit=5000)
    refs = sorted({r["referentiecode_rdw"] for r in types if model_matches(r.get("type", ""), model)})
    if not refs:
        return []
    out = []
    for i in range(0, len(refs), 100):
        chunk = refs[i:i + 100]
        rows = _q(RECALLS, where=f"referentiecode_rdw in ({_quote(chunk)})", limit=1000)
        risk = {r["referentiecode_rdw"]: r for r in _q(RISK, where=f"referentiecode_rdw in ({_quote(chunk)})", limit=1000)}
        for r in rows:
            pub = r.get("publicatiedatum_rdw") or ""
            published = f"{pub[:4]}-{pub[4:6]}-{pub[6:8]}" if len(pub) == 8 else None
            if since and published and published < since.isoformat():
                continue
            danger = risk.get(r["referentiecode_rdw"], {}).get("code_mogelijk_gevaar", "")
            out.append({
                "ref": r["referentiecode_rdw"],
                "date": published,
                "category": category_en((r.get("categorie_defect") or "").split(" - ")[-1]),
                "defect": r.get("omschrijving_defect") or "",
                "consequence": r.get("materi_le_gevolgen") or "",
                "remedy": r.get("beschrijving_van_het_herstel") or "",
                "risk": RISK_LEVEL.get(r.get("risicobeoordeling_rdw", ""), "unknown"),
                "danger_code": danger, "danger": DANGER.get(danger, ""),
                "vehicles_nl": r.get("nationaal_opgegeven_aantal_voertuigen_terugroepactie"),
                "vehicles_total": r.get("totaal_aantal_voertuigen_terugroepactie"),
                "make": make, "model": model,
            })
    return sorted(out, key=lambda r: r["date"] or "", reverse=True)


def recall_severity(r: dict) -> str:
    """Rule-based floor: RDW's own 'serious' risk assessment, or a fire/injury danger, is High."""
    if r.get("risk") == "serious" or r.get("danger_code") in ("BRA", "ONG"):
        return "High"
    return "Medium"


def clean_plate(raw: str) -> str | None:
    p = re.sub(r"[\s\-]", "", str(raw)).upper()
    return p if PLATE_RE.match(p) and not p.isdigit() and not p.isalpha() else None


def check_plates(plates: list[str]) -> list[dict]:
    """Per licence plate: make, model, first registration and the status of every recall on that car."""
    out = []
    for i in range(0, len(plates), 100):
        chunk = plates[i:i + 100]
        reg = {r["kenteken"]: r for r in _q(REGISTER, where=f"kenteken in ({_quote(chunk)})",
                                            select="kenteken,merk,handelsbenaming,datum_eerste_toelating", limit=1000)}
        status = {}
        for s in _q(STATUS, where=f"kenteken in ({_quote(chunk)})", limit=5000):
            status.setdefault(s["kenteken"], []).append(s)
        for p in chunk:
            r = reg.get(p)
            st = status.get(p, [])
            first = (r or {}).get("datum_eerste_toelating", "")
            out.append({
                "plate": p, "found": bool(r),
                "make": (r or {}).get("merk", "").title(), "model": (r or {}).get("handelsbenaming", ""),
                "first_registered": f"{first[:4]}-{first[4:6]}-{first[6:8]}" if len(first) == 8 else "",
                "open": sorted(s["referentiecode_rdw"] for s in st if s.get("code_status") == "O"),
                "repaired": sorted(s["referentiecode_rdw"] for s in st if s.get("code_status") == "P"),
            })
    return out


def sample_plates(make: str, model_like: str, n: int = 8, with_open_recall: bool = True) -> list[str]:
    """A few real registered plates of a model (for demos), fetched live and never stored."""
    where = f"merk='{make.upper()}' AND handelsbenaming like '{model_like.upper()}%'"
    if with_open_recall:
        where += " AND openstaande_terugroepactie_indicator='Ja'"
    return [r["kenteken"] for r in _q(REGISTER, select="kenteken", where=where, limit=n)]
