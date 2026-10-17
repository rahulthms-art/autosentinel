"""NHTSA official open data (US): recalls, owner complaints, VIN decoding. No API key needed."""
from __future__ import annotations

from datetime import datetime

from .http import get_json, post

RECALLS = "https://api.nhtsa.gov/recalls/recallsByVehicle"
COMPLAINTS = "https://api.nhtsa.gov/complaints/complaintsByVehicle"
VPIC_BATCH = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVINValuesBatch/"


def _date(s: str | None, fmt: str) -> str | None:
    try:
        return datetime.strptime(s, fmt).date().isoformat() if s else None
    except ValueError:
        return None


def recalls(make: str, model: str, year: int) -> list[dict]:
    data = get_json(RECALLS, {"make": make, "model": model, "modelYear": year})
    out = []
    for r in data.get("results", []):
        out.append({
            "campaign": r.get("NHTSACampaignNumber"),
            "date": _date(r.get("ReportReceivedDate"), "%d/%m/%Y"),
            "component": r.get("Component", ""),
            "summary": r.get("Summary", ""),
            "consequence": r.get("Consequence", ""),
            "remedy": r.get("Remedy", ""),
            "park_it": bool(r.get("parkIt")),
            "park_outside": bool(r.get("parkOutSide")),
            "ota": bool(r.get("overTheAirUpdate")),
            "make": make, "model": model, "year": year,
        })
    return out


def complaints(make: str, model: str, year: int) -> list[dict]:
    data = get_json(COMPLAINTS, {"make": make, "model": model, "modelYear": year})
    out = []
    for c in data.get("results", []):
        out.append({
            "odi": c.get("odiNumber"),
            "filed": _date(c.get("dateComplaintFiled"), "%m/%d/%Y"),
            "components": [x.strip() for x in (c.get("components") or "").split(",") if x.strip()],
            "summary": c.get("summary", ""),
            "crash": c.get("crash") in (True, "True"),
            "fire": c.get("fire") in (True, "True"),
            "injuries": int(c.get("numberOfInjuries") or 0),
            "deaths": int(c.get("numberOfDeaths") or 0),
            "year": year,
        })
    return out


def decode_vins(vins: list[str]) -> list[dict]:
    """Batch-decode VINs (vPIC accepts up to 50 per call)."""
    out = []
    for i in range(0, len(vins), 50):
        chunk = vins[i:i + 50]
        data = post(VPIC_BATCH, data={"DATA": ";".join(chunk), "format": "json"})
        for r in data.get("Results", []):
            out.append({
                "vin": r.get("VIN"),
                "make": (r.get("Make") or "").title(),
                "model": r.get("Model") or "",
                "year": int(r["ModelYear"]) if (r.get("ModelYear") or "").isdigit() else None,
                "plant_country": r.get("PlantCountry") or "",
                "decode_error": r.get("ErrorCode") not in ("0", None) and r.get("ErrorText") or "",
            })
    return out


def recall_severity(r: dict) -> str:
    """Rule-based floor for severity: NHTSA's own do-not-drive / park-outside flags are always High."""
    if r.get("park_it") or r.get("park_outside"):
        return "High"
    # Nearly every recall consequence mentions "risk of a crash", so only safety-critical
    # systems and fire risk set a High floor; everything else is Medium and the model decides.
    component = r.get("component", "").upper()
    if any(k in component for k in ("AIR BAG", "BRAKE", "STEERING", "FUEL SYSTEM")):
        return "High"
    if "FIRE" in r.get("consequence", "").upper():
        return "High"
    return "Medium"
