"""The Radar run: collect → de-duplicate → triage → detect signals → brief → store → alert."""
from __future__ import annotations

import json
import logging
import re
from datetime import date
from typing import Callable

from . import alerts
from .config import EXCLUDE_DOMAINS, GLOBAL_DOMAINS, MARKET_DOMAINS, SUPPLIER_DOMAINS, TRADE_PRESS, Settings
from .db import DB
from .llm import Router, parse_json
from .signals import detect_spikes, recall_data_suspect
from .sources import nhtsa, rdw, web

log = logging.getLogger("radar")
SEV_RANK = {"High": 0, "Medium": 1, "Low": 2}

PLANNER = """You plan web research for an automotive compliance & quality team.
Given a watch profile, write 6-10 precise search queries covering the requested topics and markets:
- Recalls: safety recalls/campaigns for the watched vehicles and components outside the US and the Netherlands
  (EU Safety Gate, UK DVSA, KBA, Transport Canada...). US recalls (NHTSA) and Dutch recalls (RDW) are already
  covered by official APIs.
- Regulations: new/changed rules and deadlines (UNECE WP.29 UN R-series, EU type-approval, GSR2, Euro 7,
  CO2 fleet targets, R155/R156 cybersecurity, EU Battery Regulation, Data Act...) relevant to the profile.
- Supplier risk: quality problems, shortages, insolvencies, plant fires, sanctions for the watched suppliers/components.
Write at most 10 queries, each under 15 words, as plain natural-language search phrases:
no boolean operators (AND, OR, NOT), no quotes, no minus signs.
"market" must be exactly one of: EU, UK, NL, US, Canada, Japan, Australia, Global.
Return {"queries": [{"query": "...", "topic": "Recalls|Regulations|Supplier risk", "market": "EU"}]}"""

TRIAGE = """You triage items for an automotive compliance & quality team. For EACH numbered item decide,
strictly against the watch profile:
- score: 0-10 relevance (0 = unrelated or generic news, 10 = directly affects watched vehicles/components/markets).
  Road vehicles only: items about other industries (aviation, rail, marine, consumer electronics, appliances,
  toys, food) score 0-1 even if they mention batteries, recalls or safety. Vehicles outside the profile score
  low unless they share a watched component, supplier or regulation. Road-vehicle regulations, standards and
  test protocols in the watched markets (EU/UN/national rules, Euro NCAP, battery and cybersecurity rules) and
  supplier or raw-material news affecting watched components are relevant (score 5+) even if no model is named.
- type: recall | regulation | supplier | other
- severity: High (safety risk, legal deadline < 6 months, stop-sale/production risk) | Medium | Low
- affected: which watched models / markets / components / suppliers, short
- action: one concrete recommended action for the team, short
- summary: one factual sentence
- ref: the first 3 words of the item's title, copied exactly (used to check alignment)
Every item gets a verdict, including irrelevant ones (score 0, type other, severity Low).
Some items are in Dutch (RDW): always write affected, action and summary in English.
Return {"items": [{"id": 1, "ref": "", "score": 0, "type": "", "severity": "", "affected": "", "action": "", "summary": ""}]}"""

SIGNAL_ANALYST = """You are a vehicle safety defect investigator. Below are recent owner complaints that make up a
statistical spike for one component with NO matching recall yet. Identify the likely common failure mode(s),
safety risk (crash/fire/injury potential), how strong the evidence is, and what the OEM quality team should do now.
Answer in <=150 words of Markdown, with a bold one-line verdict first. Do not invent facts beyond the complaints."""

BRIEF = """You are a senior automotive regulatory & quality analyst. Write a decision-ready Markdown risk brief
for the watch profile using ONLY the supplied findings and signals. Cite findings as [F<id>] and signals as [S<n>].
Never invent recall numbers, dates, or regulation clauses. Say plainly when evidence is thin.

## Executive summary
3-5 bullets: what is new since the last run, why it matters, what to do this week.
## Top risks
Markdown table: | Severity | Item | Affected | Deadline / timing | Action | Ref | (High first, max 10 rows)
## Early-warning signals
Complaint spikes without a recall yet, and what they suggest.
## Watch next (90 days)
"""


def _profile_text(cfg: dict) -> str:
    vehicles = "; ".join(f"{v['make']} {v['model']} {min(v['years'])}-{max(v['years'])}"
                         for v in cfg.get("vehicles", []) if v.get("years"))
    return "\n".join([
        f"Role: {cfg.get('role', '')}",
        f"Vehicles: {vehicles}",
        f"Markets: {', '.join(cfg.get('markets', []))}",
        f"Components: {cfg.get('components', '')}",
        f"Suppliers: {cfg.get('suppliers', '')}",
        f"Topics: {', '.join(cfg.get('topics', []))}",
    ])


def collect_nhtsa(cfg: dict, progress) -> tuple[list[dict], dict]:
    """US recalls (as findings) and complaints (for signals) from NHTSA."""
    findings: dict[str, dict] = {}
    per_vehicle: dict[str, dict] = {}
    for v in cfg.get("vehicles", []):
        name = f"{v['make']} {v['model']}"
        bucket = per_vehicle.setdefault(name, {"recalls": [], "complaints": []})
        for year in v.get("years", []):
            progress(f"NHTSA: {name} {year}")
            try:
                recs = nhtsa.recalls(v["make"], v["model"], year)
                comps = nhtsa.complaints(v["make"], v["model"], year)
            except Exception as e:  # one bad model-year must not kill the run
                log.warning("NHTSA failed for %s %s: %s", name, year, e)
                continue
            bucket["recalls"] += recs
            bucket["complaints"] += comps
            for r in recs:
                key = f"nhtsa:{r['campaign']}"
                f = findings.get(key)
                if f:
                    f["raw"]["years"].append(year)
                    continue
                findings[key] = {
                    "key": key, "source": "nhtsa", "type": "recall", "market": "US",
                    "title": f"NHTSA {r['campaign']}: {name}: {r['component'].split(':')[-1].title()}",
                    "url": f"https://www.nhtsa.gov/recalls?nhtsaId={r['campaign']}",
                    "summary": r["summary"], "published": r["date"],
                    "severity": nhtsa.recall_severity(r),
                    "raw": {**r, "years": [year]},
                }
    return list(findings.values()), per_vehicle


def collect_rdw(cfg: dict, progress) -> list[dict]:
    """Official Dutch recalls (RDW register) for the watched vehicles, since their first model year."""
    findings = {}
    for v in cfg.get("vehicles", []):
        years = v.get("years") or [2000]
        progress(f"RDW: {v['make']} {v['model']}")
        try:
            recs = rdw.recalls(v["make"], v["model"], since=date(min(years), 1, 1))
        except Exception as e:
            log.warning("RDW failed for %s %s: %s", v["make"], v["model"], e)
            continue
        for r in recs:
            findings.setdefault(f"rdw:{r['ref']}", {
                "key": f"rdw:{r['ref']}", "source": "rdw", "type": "recall", "market": "NL",
                "title": f"RDW {r['ref']}: {v['make']} {v['model']}: {r['category'].capitalize()}",
                "url": f"https://opendata.rdw.nl/resource/j9yg-7rg9.json?referentiecode_rdw={r['ref']}",
                "summary": f"{r['defect']} Gevolg: {r['consequence']} Herstel: {r['remedy']}"[:1400],
                "published": r["date"], "severity": rdw.recall_severity(r), "raw": r,
            })
    return list(findings.values())


VALID_MARKETS = {"EU", "UK", "NL", "US", "Canada", "Japan", "Australia", "Global"}


def clean_query(q: str) -> str:
    """Search engines read boolean operators literally: drop them (and NOT-ed terms) even if the model adds them."""
    q = re.sub(r"\bNOT\s+\S+", " ", q)
    q = re.sub(r"\b(AND|OR)\b|[\"()]|(?<!\w)-\w+", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def clean_market(m) -> str:
    for part in re.split(r"[|/,\s]+", str(m or "")):
        if part in VALID_MARKETS:
            return part
    return "Global"


def fallback_queries(cfg: dict) -> list[dict]:
    """Plain, deterministic searches used when the model's plan is unusable."""
    year = date.today().year
    qs = []
    for v in cfg.get("vehicles", [])[:4]:
        qs.append({"query": f"{v['make']} {v['model']} recall Europe {year}", "topic": "Recalls", "market": "EU"})
    if "Regulations" in cfg.get("topics", []):
        qs.append({"query": f"EU vehicle type-approval regulation update {year}", "topic": "Regulations", "market": "EU"})
        qs.append({"query": f"UNECE WP.29 new UN regulation amendments {year}", "topic": "Regulations", "market": "EU"})
    if "Supplier risk" in cfg.get("topics", []) and cfg.get("components"):
        qs.append({"query": f"automotive supplier shortage {cfg['components'].split(',')[0]} {year}",
                   "topic": "Supplier risk", "market": "Global"})
    return qs


def collect_web(router: Router, settings: Settings, cfg: dict, progress, errors: list | None = None) -> list[dict]:
    def plan(thinking=None):
        text = router.chat("fast", "Plan web research", PLANNER + "\n\nRespond with valid JSON only.",
                           _profile_text(cfg), temperature=0.0, max_tokens=4096, thinking=thinking)
        p = parse_json(text, None)
        qs = p.get("queries", []) if isinstance(p, dict) else p
        if not isinstance(qs, list) or not qs:  # cut-off or malformed JSON: salvage the complete query objects
            qs = [{"query": m.group(1), "topic": m.group(2), "market": m.group(3)} for m in re.finditer(
                r'"query"\s*:\s*"([^"]{3,200})"\s*,\s*"topic"\s*:\s*"([^"]*)"\s*,\s*"market"\s*:\s*"([^"]*)"', text)]
        return [q if isinstance(q, dict) else {"query": q} for q in qs if q][:10]

    queries = plan() or plan(thinking=True)  # an unreadable plan must not silently skip all web research
    if not queries:
        queries = fallback_queries(cfg)
        progress("Search plan unavailable: using standard queries")
    out: dict[str, dict] = {}
    for q in queries:
        text = clean_query(q.get("query") or "") if isinstance(q.get("query"), str) else ""
        if not text:
            continue
        market = clean_market(q.get("market"))
        topic = q.get("topic") or ""
        domains = SUPPLIER_DOMAINS if topic == "Supplier risk" else MARKET_DOMAINS.get(market, GLOBAL_DOMAINS)
        progress(f"Tavily: {text}")
        try:
            results = web.search(settings.tavily_api_key, text, max_results=5, time_range=cfg.get("time_range"),
                                 include_domains=domains, exclude_domains=EXCLUDE_DOMAINS)
            if not results and domains is not TRADE_PRESS:  # nothing official: fall back to trade press only
                results = web.search(settings.tavily_api_key, text, max_results=5, time_range=cfg.get("time_range"),
                                     include_domains=TRADE_PRESS, exclude_domains=EXCLUDE_DOMAINS)
        except Exception as e:
            log.warning("Tavily failed for %r: %s", text, e)
            if errors is not None:
                errors.append(f"{text}: {type(e).__name__}")
            continue
        for r in results:
            key = f"web:{r['url']}"
            out.setdefault(key, {
                "key": key, "source": "web", "type": "other", "market": market,
                "title": r["title"], "url": r["url"], "summary": r["content"],
                "published": r.get("published"), "raw": {"query": text, "topic": topic},
            })
    return list(out.values())


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _aligned(ref: str, title: str) -> bool:
    """The model's echoed title words must appear among the first words of the item's title."""
    ref_w, head = _words(ref)[:3], set(_words(title)[:8])
    return bool(ref_w) and sum(w in head for w in ref_w) >= min(2, len(ref_w))


def _triage_batch(router: Router, profile: str, chunk: list[dict], step: str, thinking: bool | None = None) -> dict:
    listing = "\n\n".join(f"[{n}] ({it['source']}) {it['title']}\n{(it['summary'] or '')[:700]}"
                          for n, it in enumerate(chunk, 1))
    res = router.json("fast", step, TRIAGE, f"Watch profile:\n{profile}\n\nItems:\n{listing}",
                      default={"items": []}, max_tokens=4096, thinking=thinking)
    rows = res.get("items", []) if isinstance(res, dict) else res
    out = {}
    for r in rows if isinstance(rows, list) else []:
        # A verdict needs an id and a numeric score; "affected" may legitimately be empty for irrelevant items
        if not isinstance(r, dict):
            continue
        try:  # models sometimes emit numbers as strings ("7")
            float(r.get("score"))
            rid = int(r.get("id"))
        except (TypeError, ValueError):
            continue
        # Reject degenerate verdicts (blank type/severity) and verdicts attached to the wrong item
        if (1 <= rid <= len(chunk) and r.get("type") in ("recall", "regulation", "supplier", "other")
                and r.get("severity") in SEV_RANK and _aligned(r.get("ref", ""), chunk[rid - 1]["title"])):
            out[id(chunk[rid - 1])] = r
    return out


def triage(router: Router, cfg: dict, items: list[dict], progress, batch: int = 10) -> None:
    """Score/classify items in place with the fast model, in batches.
    Every item must come back with a verdict; items the model skipped are retried once in smaller batches."""
    profile = _profile_text(cfg)
    verdicts: dict[int, dict] = {}
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        progress(f"Triage {i + 1}-{i + len(chunk)} of {len(items)}")
        verdicts.update(_triage_batch(router, profile, chunk, "Triage findings"))
    # Retry what the model skipped or got wrong: smaller batches first, then with reasoning on
    for attempt, thinking in (("Triage retry", None), ("Triage retry (reasoning)", True)):
        missing = [it for it in items if id(it) not in verdicts]
        if not missing:
            break
        progress(f"{attempt}: {len(missing)} item(s)")
        for i in range(0, len(missing), 5):
            verdicts.update(_triage_batch(router, profile, missing[i:i + 5], attempt, thinking=thinking))
    for it in items:
        r = verdicts.get(id(it), {})
        try:
            it["score"] = float(r.get("score", 5))
        except (TypeError, ValueError):
            it["score"] = 5.0
        if it["source"] == "web" and r.get("type") in ("recall", "regulation", "supplier", "other"):
            it["type"] = r["type"]
        llm_sev = r.get("severity") if r.get("severity") in SEV_RANK else "Medium"
        # Rule-based severity (e.g. NHTSA park-it) is a floor the model cannot lower.
        floor = it.get("severity")
        it["severity"] = min([s for s in (floor, llm_sev) if s], key=SEV_RANK.get)
        it["affected"] = str(r.get("affected", ""))[:300]
        it["action"] = str(r.get("action", ""))[:300]
        if it["source"] in ("web", "rdw") and r.get("summary"):  # RDW texts are Dutch: keep the English summary
            it["summary"] = str(r["summary"])[:600]
        if it["score"] < 4 and it["source"] == "web":
            it["status"] = "dismissed"  # stored so it is never re-triaged, hidden by default


def analyse_signals(router: Router, per_vehicle: dict, progress, max_analyses: int = 3) -> list[dict]:
    signals = []
    for vehicle, data in per_vehicle.items():
        if recall_data_suspect(data["complaints"], data["recalls"]):
            log.warning("Skipping signals for %s: complaints found but no recalls (check model name)", vehicle)
            progress(f"Skipped early warning for {vehicle}: recall lookup returned nothing (check model name)")
            continue
        for s in detect_spikes(data["complaints"], data["recalls"]):
            signals.append({**s, "vehicle": vehicle})
    for s in [s for s in signals if not s["recalled"]][:max_analyses]:
        progress(f"Analysing signal: {s['vehicle']}: {s['component']}")
        s["analysis"] = router.chat(
            "deep", "Analyse complaint spike", SIGNAL_ANALYST,
            f"Vehicle: {s['vehicle']}\nComponent: {s['component']}\nRecent complaints: {s['recent']} "
            f"(baseline {s['baseline']} per 90 days, x{s['ratio']})\nSeverity: {json.dumps(s['severity'])}\n\n"
            + "\n---\n".join(s["examples"]))
    return signals


def write_brief(router: Router, cfg: dict, new_items: list[dict], signals: list[dict], open_count: int) -> str:
    relevant = [f for f in new_items if f.get("status") != "dismissed"]
    relevant.sort(key=lambda f: (SEV_RANK.get(f.get("severity"), 3), -(f.get("score") or 0)))
    if not relevant and not signals:
        return f"## No new items\nNothing new since the last run. {open_count} items remain open."
    findings_txt = "\n\n".join(
        f"[F{f['id']}] {f['severity']} | {f['type']} | {f['market']} | {f['title']}\n"
        f"Affected: {f.get('affected')}\nAction: {f.get('action')}\nPublished: {f.get('published')}\n"
        f"{(f.get('summary') or '')[:500]}\nURL: {f.get('url')}" for f in relevant[:25])
    signals_txt = "\n\n".join(
        f"[S{n}] {s['priority']} | {s['vehicle']}: {s['component']}: {s['recent']} complaints in 90d vs baseline "
        f"{s['baseline']} (x{s['ratio']}); recall exists: {s['recalled']}\n{s.get('analysis') or ''}"
        for n, s in enumerate(signals[:8], 1))
    return router.chat("deep", "Write risk brief", BRIEF, max_tokens=8192, user=
                       f"Today: {date.today()}\nWatch profile:\n{_profile_text(cfg)}\n"
                       f"Open items from earlier runs: {open_count}\n\nNEW FINDINGS:\n{findings_txt or 'none'}"
                       f"\n\nSIGNALS:\n{signals_txt or 'none'}")


def run_watchlist(db: DB, router: Router, settings: Settings, wl_id: int,
                  progress: Callable[[str], None] = lambda m: None) -> dict:
    wl = db.watchlist(wl_id)
    if not wl:
        raise ValueError(f"Watchlist {wl_id} not found")
    cfg = wl["config"]
    topics = cfg.get("topics", [])
    run_id = db.start_run(wl_id)
    router.on_call = lambda rec: db.log_llm(run_id, rec)
    try:
        known = db.known_keys(wl_id)
        items, per_vehicle, web_errors, web_count = [], {}, [], 0
        if "Recalls" in topics or "Early warning" in topics:
            nh_items, per_vehicle = collect_nhtsa(cfg, progress)
            if "Recalls" in topics:
                items += nh_items
        if "Recalls" in topics and set(cfg.get("markets", [])) & {"NL", "EU"}:
            items += collect_rdw(cfg, progress)
        if settings.tavily_api_key and set(topics) & {"Recalls", "Regulations", "Supplier risk"}:
            web_items = collect_web(router, settings, cfg, progress, errors=web_errors)
            web_count = len(web_items)
            items += web_items
        new_items = [it for it in items if it["key"] not in known]
        progress(f"{len(items)} items collected, {len(new_items)} new")

        if new_items:
            triage(router, cfg, new_items, progress)
        db.add_findings(wl_id, run_id, new_items)
        stored = {f["key"]: f for f in db.findings(wl_id, run_id)}
        for it in new_items:
            it["id"] = stored.get(it["key"], {}).get("id")

        signals = analyse_signals(router, per_vehicle, progress) if "Early warning" in topics else []
        db.add_signals(run_id, wl_id, signals)

        open_count = sum(1 for f in db.findings(wl_id)
                         if f["status"] in ("new", "acknowledged") and f["first_run_id"] != run_id)
        progress("Writing risk brief")
        brief = write_brief(router, cfg, new_items, signals, open_count)

        relevant = [f for f in new_items if f.get("status") != "dismissed"]
        stats = {
            "collected": len(items), "new": len(new_items), "relevant": len(relevant),
            "high": sum(f["severity"] == "High" for f in relevant),
            "signals": len(signals), "unrecalled_signals": sum(not s["recalled"] for s in signals),
            "priority_signals": sum(not s["recalled"] and s["priority"] == "Priority" for s in signals),
            "web_search_errors": len(web_errors), "web_results": web_count,
            "llm_calls": len(router.trace),
            "tokens": sum((c.get("prompt_tokens") or 0) + (c.get("completion_tokens") or 0) for c in router.trace),
        }
        db.finish_run(run_id, "ok", stats, brief)
        if settings.alert_webhook and (stats["high"] or stats["priority_signals"]):
            alerts.send(settings.alert_webhook, wl["name"], stats, relevant, signals)
        return {"run_id": run_id, "stats": stats, "brief": brief}
    except Exception as e:
        db.finish_run(run_id, "error", error=str(e)[:1000])
        raise
