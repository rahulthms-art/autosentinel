"""Fleet exposure check: which vehicles in a VIN list are covered by recalls."""
from __future__ import annotations

import re

from .sources import nhtsa

VIN_RE = re.compile(r"^[A-HJ-NPR-Z0-9]{17}$")


def clean_vins(raw: list[str]) -> tuple[list[str], list[str]]:
    good, bad = [], []
    for v in raw:
        v = str(v).strip().upper()
        if not v:
            continue
        (good if VIN_RE.match(v) else bad).append(v)
    return list(dict.fromkeys(good)), bad


def check_fleet(vins: list[str], progress=lambda m: None) -> list[dict]:
    progress(f"Decoding {len(vins)} VINs")
    decoded = nhtsa.decode_vins(vins)
    recall_cache: dict[tuple, list[dict]] = {}
    rows = []
    for d in decoded:
        key = (d["make"], d["model"], d["year"])
        if d["make"] and d["model"] and d["year"] and key not in recall_cache:
            progress(f"Recalls: {d['make']} {d['model']} {d['year']}")
            try:
                recall_cache[key] = nhtsa.recalls(*key)
            except Exception:
                recall_cache[key] = []
        recs = recall_cache.get(key, [])
        rows.append({
            "VIN": d["vin"], "Make": d["make"], "Model": d["model"], "Year": d["year"],
            "Recalls": len(recs),
            "Do-not-drive / park outside": any(r["park_it"] or r["park_outside"] for r in recs),
            "Campaigns": ", ".join(r["campaign"] for r in recs),
            "Components": "; ".join(sorted({r["component"].split(":")[0].title() for r in recs})),
            "Decode note": d["decode_error"][:80] if d["decode_error"] else "",
        })
    return rows
