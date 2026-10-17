"""Demo snapshot: package a watchlist's real results so a fresh public demo shows them instantly,
without spending AI credits. Imported model calls are tagged and excluded from the budget.

Export:  python -m radar.snapshot export <watchlist_id> demo/snapshot.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from .db import DB

TAG = "[snapshot] "


def export(db: DB, wl_id: int) -> dict:
    wl = db.watchlist(wl_id)
    runs = [r for r in db.runs(wl_id, limit=1000) if r["status"] == "ok"]
    run_ids = [r["id"] for r in runs]
    return {
        "watchlist": {"name": wl["name"], "config": wl["config"]},
        "runs": runs,
        "findings": db.q("SELECT * FROM findings WHERE watchlist_id=?", (wl_id,)),
        "signals": db.q("SELECT * FROM signals WHERE watchlist_id=?", (wl_id,)),
        "llm_calls": [dict(c, run_id=rid) for rid in run_ids
                      for c in db.q("SELECT * FROM llm_calls WHERE run_id=?", (rid,))],
    }


def import_snapshot(db: DB, data: dict) -> int:
    wl_id = db.save_watchlist(data["watchlist"]["name"] + " (demo data)", data["watchlist"]["config"])
    run_map = {}
    with db.tx() as c:
        for r in sorted(data["runs"], key=lambda r: r["id"]):
            run_map[r["id"]] = c.execute(
                "INSERT INTO runs (watchlist_id, started_at, finished_at, status, stats, brief, error) "
                "VALUES (?,?,?,?,?,?,?)", (wl_id, r["started_at"], r["finished_at"], r["status"],
                                           json.dumps(r["stats"]), r["brief"], r["error"])).lastrowid
        fcols = ["key", "source", "type", "title", "url", "summary", "published", "market", "severity", "score",
                 "affected", "action", "status", "first_seen", "raw"]
        for f in data["findings"]:
            c.execute(f"INSERT INTO findings (watchlist_id, first_run_id, {', '.join(fcols)}) "
                      f"VALUES (?,?,{','.join('?' * len(fcols))})",
                      [wl_id, run_map.get(f["first_run_id"])] + [f[k] for k in fcols])
        for s in data["signals"]:
            c.execute("INSERT INTO signals (run_id, watchlist_id, vehicle, component, recent, baseline, ratio,"
                      " recalled, analysis) VALUES (?,?,?,?,?,?,?,?,?)",
                      (run_map.get(s["run_id"]), wl_id, s["vehicle"], s["component"], s["recent"], s["baseline"],
                       s["ratio"], s["recalled"], s["analysis"]))
        for k in data["llm_calls"]:
            c.execute("INSERT INTO llm_calls (run_id, at, step, model, seconds, prompt_tokens, completion_tokens,"
                      " ok, error) VALUES (?,?,?,?,?,?,?,?,?)",
                      (run_map.get(k["run_id"]), k["at"], TAG + (k["step"] or ""), k["model"], k["seconds"],
                       k["prompt_tokens"], k["completion_tokens"], k["ok"], k["error"]))
    return wl_id


def load_if_fresh(db: DB, path: Path) -> bool:
    """Import the snapshot once, into a deployment with no successful run of its own."""
    if not path.exists() or db.q("SELECT 1 FROM runs WHERE status='ok' LIMIT 1"):
        return False
    for w in db.watchlists():  # only empty starter watchlists (and failed runs) exist: replace them
        db.delete_watchlist(w["id"])
    import_snapshot(db, json.loads(path.read_text()))
    return True


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "export":
    from .config import Settings
    out = export(DB(Settings().db_path), int(sys.argv[2]))
    Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
    Path(sys.argv[3]).write_text(json.dumps(out, default=str))
    print(f"exported {len(out['runs'])} runs, {len(out['findings'])} findings, {len(out['signals'])} signals")
