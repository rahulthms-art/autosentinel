"""Backtest the Nemotron second-stage filter on TRAIN and held-out TEST vehicles."""
from __future__ import annotations

import json
from collections import Counter

from evals.backtest import evaluate, fmt, load
from evals.llm_filter import screen_all
from evals.run_evals import settings_from_env
from radar.llm import Router, list_nvidia_models, make_client, pick_default
from radar.signals import SpikeParams

CANDIDATES = SpikeParams(recent_days=90, min_recent=3, min_ratio=1.5, max_p=1.0, volume_adjust=False)
OLD = SpikeParams(recent_days=90, min_recent=3, min_ratio=2.0, max_p=1.0, volume_adjust=False)


def key(v, as_of, s):
    return f"{v['make']}|{v['model']}|{s['component']}|{as_of.isoformat()}"


if __name__ == "__main__":
    st = settings_from_env()
    client = make_client(st.nebius_api_key, st.nebius_base_url)
    models = list_nvidia_models(client)
    router = Router(client, pick_default(models, ["nano"]), pick_default(models, ["ultra"]))
    train, test = load()
    jobs = {}

    def collect(v, as_of, s):
        jobs[key(v, as_of, s)] = (v["make"], v["model"], s)
        return False
    for vs in (train, test):
        evaluate(vs, CANDIDATES, detector=collect)
    print(f"Screening {len(jobs)} candidate alerts with {router.fast} ...", flush=True)
    decisions = screen_all(router, jobs)
    print("signal mix:", Counter(d["signal"] for k, d in decisions.items() if k in jobs))

    out = {}
    for name, vs in (("TRAIN", train), ("TEST", test)):
        out[name] = {"old_statistical": evaluate(vs, OLD), "candidates_only": evaluate(vs, CANDIDATES)}
        for level, allowed in (("nemotron_strong", {"strong"}), ("nemotron_strong_or_weak", {"strong", "weak"})):
            out[name][level] = evaluate(vs, CANDIDATES, detector=lambda v, a, s, al=allowed: decisions.get(key(v, a, s), {}).get("signal") in al)
        print(f"\n{name}")
        for k, m in out[name].items():
            print(f"  {k:26} {fmt(m)}")
    json.dump({n: {k: {x: y for x, y in m.items() if x != 'alert_list'} for k, m in d.items()} for n, d in out.items()},
              open("evals/cache/filter_backtest.json", "w"), indent=2)
