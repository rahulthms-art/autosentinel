"""Second-stage early-warning filter: Nemotron reads the complaint narratives behind a statistical
spike and judges whether they describe one consistent, safety-relevant defect.

Vehicle make/model are masked so the model cannot rely on remembering famous historical recalls.
Decisions are cached in evals/cache/llm_filter.json.
"""
from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from radar.llm import Router, parse_json

CACHE = Path(__file__).parent / "cache" / "llm_filter.json"
_lock = threading.Lock()

SCREEN = """You are a vehicle safety defect investigator screening owner complaints filed with a regulator.
A statistical spike was detected for one component. Decide whether the complaints point to ONE consistent,
specific, safety-relevant failure mode that could plausibly lead to a safety recall, as opposed to unrelated
issues, general dissatisfaction, normal wear, driver error, dealer service problems, or non-safety annoyances.
- strong: most complaints describe the same specific failure with a safety consequence
- weak: some shared pattern, but vague, mixed, or limited safety consequence
- none: no consistent defect pattern
Return {"signal": "strong|weak|none", "failure_mode": "<= 12 words", "reason": "<= 25 words"}"""


def mask(text: str, make: str, model: str) -> str:
    for word in {make, model, *model.split()}:
        if len(word) > 1:
            text = re.sub(re.escape(word), "[VEHICLE]", text, flags=re.I)
    return text


def prompt(make, model, s) -> str:
    sev = s["severity"]
    complaints = "\n".join(f"{i}. {mask(t, make, model)[:400]}" for i, t in enumerate(s["examples"][:12], 1))
    return (f"Vehicle: Vehicle A\nComponent: {s['component']}\nComplaints in last 90 days: {s['recent']} "
            f"(expected about {s['baseline']})\nCrashes: {sev['crash']}, fires: {sev['fire']}, injuries: {sev['injuries']}\n\n"
            f"Complaints:\n{complaints}")


def load_cache() -> dict:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def screen_all(router: Router, jobs: dict, workers: int = 6) -> dict:
    """jobs: {key: (make, model, spike)} -> {key: {"signal": ...}}; uses and fills the cache."""
    cache = load_cache()
    todo = [k for k in jobs if k not in cache]
    done = [0]

    def run(k):
        make, model, s = jobs[k]
        try:
            res = router.json("fast", "Screen spike", SCREEN, prompt(make, model, s), default={"signal": "none"},
                              max_tokens=300)
        except Exception as e:
            res = {"signal": "error", "reason": str(e)[:100]}
        if not isinstance(res, dict) or res.get("signal") not in ("strong", "weak", "none"):
            res = {"signal": "none", "reason": "unparseable"}
        with _lock:
            cache[k] = res
            done[0] += 1
            if done[0] % 100 == 0:
                CACHE.write_text(json.dumps(cache))
                print(f"  screened {done[0]}/{len(todo)}", flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(run, todo))
    CACHE.write_text(json.dumps(cache))
    return cache
