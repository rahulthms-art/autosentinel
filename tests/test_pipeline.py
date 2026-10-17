import json

from radar import analyst, pipeline
from radar.config import Settings

CFG = {"role": "Compliance", "vehicles": [{"make": "Acme", "model": "Volt-X", "years": [2022]}],
       "markets": ["EU"], "components": "HV battery", "suppliers": "", "time_range": "month",
       "topics": ["Recalls", "Regulations", "Early warning"]}

RECALL = {"campaign": "26V001000", "date": "2026-09-01", "component": "ELECTRICAL SYSTEM:BATTERY",
          "summary": "Battery may overheat", "consequence": "fire", "remedy": "replace",
          "park_it": False, "park_outside": True, "ota": False, "make": "Acme", "model": "Volt-X", "year": 2022}


def triage_reply(user):
    blocks = user.split("Items:\n", 1)[1].split("\n\n")
    items = {int(b[1:b.index("]")]): b for b in blocks if b.startswith("[")}
    def ref(block):  # first 3 words of the title line "[n] (source) Title"
        return " ".join(block.split("\n")[0].split(") ", 1)[1].split()[:3])
    return json.dumps({"items": [{"id": i, "ref": ref(text), "score": 2 if "celebrity" in text else 8,
                                  "type": "regulation", "severity": "Low", "affected": "Volt-X",
                                  "action": "review", "summary": "s"} for i, text in items.items()]})


RULES = [
    ("plan web research", json.dumps({"queries": [{"query": "EU battery regulation 2026", "topic": "Regulations",
                                                   "market": "EU"}]})),
    ("triage items", triage_reply),
    ("defect investigator", "**Verdict: likely wiring fault**"),
    ("risk brief", "## Executive summary\n- new battery recall [F1]"),
]


RDW_RECALL = {"ref": "MGP260096", "date": "2026-03-20", "category": "electrical system",
              "defect": "Accu kan oververhit raken.", "consequence": "Brand.", "remedy": "Software update.",
              "risk": "serious", "danger_code": "BRA", "danger": "fire with injury"}


def _patch_sources(monkeypatch, web_results, rdw_recalls=()):
    monkeypatch.setattr(pipeline.rdw, "recalls", lambda *a, **k: [dict(r) for r in rdw_recalls])
    monkeypatch.setattr(pipeline.nhtsa, "recalls", lambda *a: [dict(RECALL)])
    monkeypatch.setattr(pipeline.nhtsa, "complaints", lambda *a: [])
    monkeypatch.setattr(pipeline.web, "search", lambda *a, **k: list(web_results))


def test_run_end_to_end_and_dedup(db, make_router, monkeypatch):
    web_results = [
        {"title": "EU Battery Regulation due-diligence delayed", "url": "https://ec.europa.eu/a",
         "content": "battery passport", "published": None, "query": "q"},
        {"title": "Car celebrity gossip", "url": "https://ec.europa.eu/b", "content": "celebrity",
         "published": None, "query": "q"},
    ]
    _patch_sources(monkeypatch, web_results)
    wl_id = db.save_watchlist("t", CFG)
    settings = Settings(nebius_api_key="x", tavily_api_key="y", db_path=":memory:", alert_webhook="")

    router, fake = make_router(RULES)
    out = pipeline.run_watchlist(db, router, settings, wl_id)
    assert out["stats"]["new"] == 3 and out["stats"]["relevant"] == 2
    f = {x["key"]: x for x in db.findings(wl_id)}
    # NHTSA park-outside recall keeps High severity even though the model said Low
    assert f["nhtsa:26V001000"]["severity"] == "High"
    assert f["web:https://ec.europa.eu/b"]["status"] == "dismissed"
    # fast model did planning + triage, deep model wrote the brief
    tiers = {m for m, s in fake.calls}
    assert tiers == {"nvidia/nemotron-nano", "nvidia/nemotron-ultra"}
    assert len(db.llm_calls(out["run_id"])) == len(fake.calls)

    # Second run: same sources → nothing new, no triage call, no-news brief without the deep model
    router2, fake2 = make_router(RULES)
    out2 = pipeline.run_watchlist(db, router2, settings, wl_id)
    assert out2["stats"]["new"] == 0
    assert out2["brief"].startswith("## No new items")
    assert all("triage" not in s.lower() for _, s in fake2.calls)


def test_failed_run_is_recorded(db, make_router, monkeypatch):
    _patch_sources(monkeypatch, [])
    wl_id = db.save_watchlist("t", CFG)

    def boom(user):
        raise RuntimeError("model down")
    router, _ = make_router([("plan web research", boom)])
    settings = Settings(nebius_api_key="x", tavily_api_key="y", db_path=":memory:", alert_webhook="")
    try:
        pipeline.run_watchlist(db, router, settings, wl_id)
    except RuntimeError:
        pass
    run = db.runs(wl_id)[0]
    assert run["status"] == "error" and "model down" in run["error"]


def test_analyst_uses_tools_then_answers(db, make_router, monkeypatch):
    monkeypatch.setattr(analyst.nhtsa, "recalls", lambda *a: [dict(RECALL)])
    wl_id = db.save_watchlist("t", CFG)
    replies = iter([
        json.dumps({"thought": "check recalls", "tool": "nhtsa_recalls",
                    "args": {"make": "Acme", "model": "Volt-X", "year": 2022}}),
        json.dumps({"thought": "done", "final": "Yes: recall 26V001000 (park outside)."}),
    ])
    router, _ = make_router([("analyst agent", lambda u: next(replies))])
    steps = []
    ans = analyst.ask(db, router, Settings(tavily_api_key="y"), wl_id, "Any Volt-X battery recalls?", [],
                      on_step=steps.append)
    assert "26V001000" in ans
    assert steps[0]["tool"] == "nhtsa_recalls" and "26V001000" in steps[0]["observation"]


def test_triage_retries_items_the_model_skipped(make_router):
    calls = {"n": 0}

    def flaky(user):
        calls["n"] += 1
        if calls["n"] == 1:
            return json.dumps({"items": [{"id": 1, "ref": "t0", "score": 1, "type": "other", "severity": "Low",
                                          "affected": "", "action": ""}]})
        return triage_reply(user)  # retry returns full verdicts

    router, _ = make_router([("triage items", flaky)])
    items = [{"source": "web", "title": f"t{i}", "summary": "s", "severity": None} for i in range(3)]
    pipeline.triage(router, CFG, items, lambda m: None)
    assert items[0]["score"] == 1 and items[0]["status"] == "dismissed"  # empty "affected" is a valid verdict
    assert all(it["affected"] for it in items[1:])
    assert calls["n"] == 2  # one batch + one retry for the 2 skipped items


def test_triage_rejects_misaligned_and_degenerate_verdicts(make_router):
    def bad_then_good(user):
        if "reasoning" in bad_then_good.step:
            return triage_reply(user)
        # wrong ref (verdict for another item) and a blank "lazy" verdict
        return json.dumps({"items": [{"id": 1, "ref": "completely other title", "score": 10, "type": "recall",
                                      "severity": "High", "affected": "x"},
                                     {"id": 2, "ref": "t1", "score": 0, "type": "", "severity": "", "affected": ""}]})
    bad_then_good.step = ""
    router, fake = make_router([("triage items", bad_then_good)])
    orig = router.json
    def spy(tier, step, *a, **k):
        bad_then_good.step = step
        return orig(tier, step, *a, **k)
    router.json = spy
    items = [{"source": "web", "title": f"t{i}", "summary": "s", "severity": None} for i in range(2)]
    pipeline.triage(router, CFG, items, lambda m: None)
    assert all(it["score"] == 8 for it in items)  # both came from the final reasoning retry


def test_rdw_recalls_become_findings_with_english_summary(db, make_router, monkeypatch):
    _patch_sources(monkeypatch, [], rdw_recalls=[RDW_RECALL])
    wl_id = db.save_watchlist("t", {**CFG, "markets": ["NL"]})
    settings = Settings(nebius_api_key="x", tavily_api_key="y", db_path=":memory:", alert_webhook="")
    router, _ = make_router(RULES)
    pipeline.run_watchlist(db, router, settings, wl_id)
    f = {x["key"]: x for x in db.findings(wl_id)}["rdw:MGP260096"]
    assert f["market"] == "NL" and f["type"] == "recall" and f["severity"] == "High"  # RDW 'serious' floor
    assert f["summary"] == "s"  # Dutch text replaced by the model's English summary
    assert "Electrical system" in f["title"]


def test_unreadable_search_plan_falls_back_to_standard_queries(make_router, monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline.web, "search", lambda key, q, **k: seen.append(q) or [])
    router, _ = make_router([("plan web research", "not json at all")])
    settings = Settings(tavily_api_key="y")
    pipeline.collect_web(router, settings, CFG, lambda m: None)
    assert seen and any("Leaf" in q or "Volt-X" in q for q in seen)  # standard per-vehicle queries ran


def test_cut_off_search_plan_is_salvaged(make_router, monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline.web, "search", lambda key, q, **k: seen.append(q) or [])
    truncated = ('{"queries": [{"query": "EU battery regulation 2026", "topic": "Regulations", "market": "EU"}, '
                 '{"query": "UNECE R155 amendment", "topic": "Regulations", "market": "EU"}, {"query": "Tesla Mod')
    router, _ = make_router([("plan web research", truncated)])
    pipeline.collect_web(router, Settings(tavily_api_key="y"), CFG, lambda m: None)
    assert list(dict.fromkeys(seen)) == ["EU battery regulation 2026", "UNECE R155 amendment"]  # each query also retried on trade press


def test_analyst_can_use_rdw_and_tidies_citations(db, make_router, monkeypatch):
    monkeypatch.setattr(analyst.rdw, "recalls", lambda make, model: [{
        "ref": "MGP260096", "date": "2026-03-20", "category": "airbags", "risk": "serious",
        "danger": "accident with injury", "defect": "Airbag kan niet goed werken.", "remedy": "Vervangen."}])
    wl_id = db.save_watchlist("t", CFG)
    replies = iter([
        json.dumps({"tool": "rdw_recalls", "args": {"make": "Acme", "model": "Volt-X"}}),
        json.dumps({"final": "Yes: RDW MGP260096 airbag recall【1†L1-L4】."}),
    ])
    router, _ = make_router([("analyst agent", lambda u: next(replies))])
    steps = []
    ans = analyst.ask(db, router, Settings(), wl_id, "Open airbag recalls in NL?", [], on_step=steps.append)
    assert steps[0]["tool"] == "rdw_recalls" and "MGP260096" in steps[0]["observation"]
    assert ans == "Yes: RDW MGP260096 airbag recall [1]."
