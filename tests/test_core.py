from datetime import date

from radar.fleet import clean_vins
from radar.llm import parse_json, strip_thinking
from radar.signals import detect_spikes
from radar.sources.nhtsa import recall_severity


def test_parse_json_variants():
    assert parse_json('```json\n{"a": 1}\n```', None) == {"a": 1}
    assert parse_json('Sure! {"a": [1, 2]} hope it helps', None) == {"a": [1, 2]}
    assert parse_json('["x", "y"]', None) == ["x", "y"]
    assert parse_json("no json here", {"d": 1}) == {"d": 1}
    multi = '{"tool": "web_search", "args": {"q": "a"}} {"tool": "web_search", "args": {"q": "b"}}'
    assert parse_json(multi, None) == {"tool": "web_search", "args": {"q": "a"}}
    assert strip_thinking("<think>secret</think> answer") == "answer"


def _c(day, comp, fire=False):
    return {"filed": day, "components": [comp], "summary": "s", "crash": False,
            "fire": fire, "injuries": 0, "deaths": 0}


def test_spike_detection_flags_unrecalled_spike():
    as_of = date(2026, 10, 1)
    complaints = [_c("2026-09-%02d" % d, "ELECTRICAL SYSTEM", fire=d < 3) for d in range(1, 9)]  # 8 recent
    complaints += [_c("2026-03-01", "ELECTRICAL SYSTEM")]                                      # 1 in baseline
    complaints += [_c("2026-09-05", "BRAKES"), _c("2026-02-05", "BRAKES")] * 2                  # steady
    recalls = [{"component": "BRAKES:HYDRAULIC", "date": "2026-05-01"}]
    spikes = detect_spikes(complaints, recalls, as_of=as_of)
    assert [s["component"] for s in spikes] == ["ELECTRICAL SYSTEM"]
    s = spikes[0]
    assert s["recent"] == 8 and not s["recalled"] and s["severity"]["fire"] == 2 and s["ratio"] >= 2


def test_spike_marked_recalled_when_recall_exists():
    complaints = [_c("2026-09-%02d" % d, "AIR BAGS") for d in range(1, 6)]
    spikes = detect_spikes(complaints, [{"component": "AIR BAGS:FRONTAL", "date": "2026-08-01"}],
                           as_of=date(2026, 10, 1))
    assert spikes and spikes[0]["recalled"]


def test_recall_severity_rules():
    assert recall_severity({"park_it": True, "component": "x", "consequence": ""}) == "High"
    assert recall_severity({"component": "AIR BAGS:FRONTAL", "consequence": ""}) == "High"
    assert recall_severity({"component": "LATCHES", "consequence": "door may open"}) == "Medium"
    assert recall_severity({"component": "BACK OVER PREVENTION", "consequence": "increasing the risk of a crash"}) == "Medium"
    assert recall_severity({"component": "ELECTRICAL SYSTEM", "consequence": "increasing the risk of a fire"}) == "High"


def test_clean_vins():
    good, bad = clean_vins(["5n1at2mv0lc700000", "5N1AT2MV0LC700000", "BAD", "", "IOQ00000000000000"])
    assert good == ["5N1AT2MV0LC700000"]
    assert bad == ["BAD", "IOQ00000000000000"]


def test_recall_data_suspect_guard():
    from radar.signals import recall_data_suspect
    complaints = [_c("2026-09-01", "BRAKES")] * 60
    assert recall_data_suspect(complaints, [])
    assert not recall_data_suspect(complaints, [{"component": "BRAKES", "date": "2026-01-01"}])
    assert not recall_data_suspect(complaints[:10], [])


def test_component_families_match_complaints_and_recalls():
    from radar.signals import component_family, complaint_families, recall_families
    # complaint lists split on commas, which breaks "SERVICE BRAKES, HYDRAULIC" into two parts
    assert complaint_families({"components": ["SERVICE BRAKES", " HYDRAULIC", "ENGINE"]}) == {"SERVICE BRAKES", "ENGINE"}
    assert component_family("SERVICE BRAKES, HYDRAULIC:FOUNDATION COMPONENTS:DISC:CALIPER") == "SERVICE BRAKES"
    assert component_family("FUEL SYSTEM, GASOLINE:DELIVERY") == component_family("FUEL/PROPULSION SYSTEM")
    assert component_family("UNKNOWN OR OTHER") is None
    assert "ADAS" in recall_families("ELECTRICAL SYSTEM:ADAS:AUTONOMOUS/SELF DRIVING:SOFTWARE")
    assert component_family("LANE DEPARTURE") == component_family("FORWARD COLLISION AVOIDANCE") == "ADAS"


def test_brake_spike_is_covered_by_hydraulic_brake_recall():
    complaints = [_c("2026-09-%02d" % d, "SERVICE BRAKES") for d in range(1, 9)]
    recalls = [{"component": "SERVICE BRAKES, HYDRAULIC:CALIPER", "date": "2026-06-01"}]
    spikes = detect_spikes(complaints, recalls, as_of=date(2026, 10, 1))
    assert spikes and spikes[0]["component"] == "SERVICE BRAKES" and spikes[0]["recalled"]


def test_priority_tiers():
    from radar.signals import priority
    assert priority({"p_value": 0.0001, "severity": {"crash": 0, "fire": 0, "injuries": 0}}) == "Priority"
    assert priority({"p_value": 0.5, "severity": {"crash": 2, "fire": 1, "injuries": 0}}) == "Priority"
    assert priority({"p_value": 0.01, "severity": {"crash": 0, "fire": 0, "injuries": 0}}) == "Watch"
    assert priority({"p_value": 0.5, "severity": {"crash": 0, "fire": 0, "injuries": 0}}) == "Low"


def test_budget_guard_stops_calls_and_demo_limits(monkeypatch):
    import pytest
    from radar.budget import BudgetExceeded, Guard, LimitReached, call_cost
    from radar.db import DB
    db = DB(":memory:")
    g = Guard(db)
    assert call_cost("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B", 1_000_000, 1_000_000) == 0.30
    monkeypatch.setenv("BUDGET_USD", "1")
    db.log_llm(None, {"step": "x", "model": "nvidia/Nemotron-3-Ultra-550b", "seconds": 1, "ok": True,
                      "prompt_tokens": 200_000, "completion_tokens": 200_000})  # $1.00 at the cautious Ultra price
    with pytest.raises(BudgetExceeded):
        g.check_budget()
    # snapshot calls are history, not spend
    db.conn.execute("UPDATE llm_calls SET step='[snapshot] x'")
    g.check_budget()
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_RUNS_PER_DAY", "2")
    for _ in range(2):
        g.allow("run")
        g.record("run")
    with pytest.raises(LimitReached):
        g.allow("run")
    with pytest.raises(LimitReached):
        g.allow("question", session_count=5)


def test_snapshot_round_trip():
    from pathlib import Path
    from radar.db import DB
    from radar.snapshot import load_if_fresh
    db = DB(":memory:")
    assert load_if_fresh(db, Path(__file__).resolve().parents[1] / "demo" / "snapshot.json")
    wl = db.watchlists()[0]
    assert db.runs(wl["id"])[0]["stats"]["source"] == "snapshot" and db.findings(wl["id"])
    assert not load_if_fresh(db, Path(__file__).resolve().parents[1] / "demo" / "snapshot.json")  # only once


def test_interrupted_run_does_not_block_demo_snapshot():
    from pathlib import Path
    from radar.db import DB
    from radar.seed import ensure_default
    from radar.snapshot import load_if_fresh
    db = DB(":memory:")
    ensure_default(db)
    db.start_run(db.watchlists()[0]["id"])          # cut off by a restart
    assert db.mark_interrupted_runs() == 1
    assert load_if_fresh(db, Path(__file__).resolve().parents[1] / "demo" / "snapshot.json")
    assert [w["name"] for w in db.watchlists()] == ["EV segment watch (EU/UK/US)"]


def test_anonymizer_masks_brands_but_not_parts():
    from radar.anonymize import Anonymizer
    a = Anonymizer([{"make": "Tesla", "model": "Model Y"}, {"make": "Chevrolet", "model": "Bolt EV"}])
    assert a("NHTSA 21V387000: Tesla Model Y: Caliper") == "NHTSA 21V••••••: OEM A Vehicle 1: Caliper"
    assert a("Chevrolet Bolt EV brake caliper bolts may loosen") == "OEM B Vehicle 2 brake caliper bolts may loosen"
    assert a("VIN 5YJYGDEE3MF100000, see https://www.nhtsa.gov/x") == "VIN …0000, see [link hidden]"
    assert a("Chevy and Chevrolet") == "OEM B and OEM B"
    assert "Nissan" not in a("Nissan Leaf") and a("Nissan again").startswith("OEM C")
    assert a("NVIDIA Nemotron on Nebius with Tavily") == "NVIDIA Nemotron on Nebius with Tavily"  # sponsors stay
    assert Anonymizer(enabled=False)("Tesla") == "Tesla"
    assert a("Transport Canada Recall 2026445 and Safety Gate A12/01234/26") == \
        "Transport Canada Recall ••••••• and Safety Gate A••/•••••/••"


def test_legacy_demo_name_is_migrated():
    from radar.db import DB
    from radar.snapshot import migrate_legacy
    db = DB(":memory:")
    wl = db.save_watchlist("EV segment watch (demo data)", {"vehicles": []})
    rid = db.start_run(wl)
    db.finish_run(rid, "ok", {"relevant": 3})
    migrate_legacy(db)
    assert db.watchlists()[0]["name"] == "EV segment watch"
    assert db.runs(wl)[0]["stats"] == {"relevant": 3, "source": "snapshot"}
