"""Headless runner for scheduled monitoring.

Run every watchlist (or one) and push alerts. Designed for cron, GitHub Actions,
or Nebius Serverless Jobs:
    python -m jobs.run_watchlists            # all watchlists
    python -m jobs.run_watchlists --id 1     # one watchlist
Env: NEBIUS_API_KEY, TAVILY_API_KEY, NEMOTRON_FAST, NEMOTRON_DEEP, RADAR_DB, ALERT_WEBHOOK_URL
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

from radar.config import Settings
from radar.db import DB
from radar.llm import Router, list_nvidia_models, make_client, pick_default
from radar.budget import Guard
from radar.pipeline import run_watchlist
from radar.seed import ensure_default


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--id", type=int, help="watchlist id (default: all)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("radar")

    s = Settings()
    if s.missing():
        log.error("Missing env: %s", ", ".join(s.missing()))
        return 2
    client = make_client(s.nebius_api_key, s.nebius_base_url)
    if not (s.fast_model and s.deep_model):
        models = list_nvidia_models(client)
        s.fast_model = s.fast_model or pick_default(models, ["nano", "super", "nemotron"])
        s.deep_model = s.deep_model or pick_default(models, ["ultra", "super", "nemotron"])
    log.info("Models: fast=%s deep=%s", s.fast_model, s.deep_model)

    db = DB(s.db_path)
    ensure_default(db)
    targets = [w for w in db.watchlists() if args.id in (None, w["id"])]
    failures = 0
    for wl in targets:
        log.info("Running watchlist %s: %s", wl["id"], wl["name"])
        try:
            router = Router(client, s.fast_model, s.deep_model, before_call=Guard(db).check_budget)
            out = run_watchlist(db, router, s, wl["id"], progress=log.info)
            log.info("Done: %s", json.dumps(out["stats"]))
        except Exception:
            log.exception("Watchlist %s failed", wl["id"])
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
