"""Ask the Analyst: a tool-using Nemotron agent over the Radar's data and live sources.

Uses a portable JSON action protocol (no dependency on provider function-calling support):
each step the model returns either {"tool": name, "args": {...}} or {"final": "markdown answer"}.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from .config import MARKET_DOMAINS, Settings
from .db import DB
from .llm import Router, parse_json
from .signals import detect_spikes
from .sources import nhtsa, rdw, web

SYSTEM = """You are AutoSentinel's analyst agent for an automotive compliance & quality team.
Answer the user's question by calling tools, then give a concise, cited Markdown answer.

Tools:
- search_findings {"text": str}: search findings already tracked by the Radar for this watchlist
- nhtsa_recalls {"make": str, "model": str, "year": int}: official US recalls for a model year
- rdw_recalls {"make": str, "model": str}: official Dutch (RDW) recalls for a model, newest first; the RDW is an
  EU type-approval authority, so these often reflect EU-wide campaigns. Texts are Dutch: answer in English.
- complaint_spikes {"make": str, "model": str, "year": int}: owner-complaint spikes per component (early warning)
- web_search {"query": str, "market": "EU|UK|US|Global"}: live web search (official domains for the market)
- decode_vin {"vin": str}: decode a VIN to make/model/year

Reply with ONE JSON object per step and nothing else:
{"thought": "...", "tool": "<name>", "args": {...}}   or   {"thought": "...", "final": "<markdown answer>"}
Rules: use at most 5 tool calls; for the Netherlands/EU use rdw_recalls before web search; cite sources inline as
plain text (campaign numbers, RDW references, finding ids, URLs), never in 【】 brackets; never invent facts;
if tools return nothing useful, say so."""


def _tools(db: DB, settings: Settings, wl_id: int) -> dict[str, Callable[..., object]]:
    def search_findings(text=""):
        return db.search_findings(wl_id, text)

    def nhtsa_recalls(make, model, year):
        return [{k: r[k] for k in ("campaign", "date", "component", "summary", "remedy", "park_it")}
                for r in nhtsa.recalls(make, model, int(year))]

    def rdw_recalls(make, model):
        return [{k: r[k] for k in ("ref", "date", "category", "risk", "danger", "defect", "remedy")}
                for r in rdw.recalls(make, model)[:20]]

    def complaint_spikes(make, model, year):
        spikes = detect_spikes(nhtsa.complaints(make, model, int(year)), nhtsa.recalls(make, model, int(year)))
        return [{k: s[k] for k in ("component", "recent", "baseline", "ratio", "recalled", "severity")}
                | {"examples": s["examples"][:3]} for s in spikes]

    def web_search(query, market="Global"):
        return [{k: r[k] for k in ("title", "url", "content")}
                for r in web.search(settings.tavily_api_key, query, max_results=5,
                                    include_domains=MARKET_DOMAINS.get(market))]

    def decode_vin(vin):
        return nhtsa.decode_vins([vin])

    return {f.__name__: f for f in (search_findings, nhtsa_recalls, rdw_recalls, complaint_spikes, web_search,
                                    decode_vin)}


def ask(db: DB, router: Router, settings: Settings, wl_id: int, question: str, history: list[dict],
        max_steps: int = 6, on_step: Callable[[dict], None] = lambda s: None) -> str:
    tools = _tools(db, settings, wl_id)
    wl = db.watchlist(wl_id)
    context = f"Watchlist: {wl['name']}\nConfig: {json.dumps(wl['config'])}\n" if wl else ""
    convo = "\n".join(f"{m['role'].upper()}: {m['content'][:1500]}" for m in history[-6:])
    scratch = ""
    for step in range(max_steps):
        reply = router.json("deep", f"Analyst step {step + 1}", SYSTEM,
                            f"{context}\nConversation:\n{convo}\n\nQUESTION: {question}\n\nPrevious steps:\n"
                            f"{scratch or 'none'}", default={"final": None})
        if not isinstance(reply, dict):
            reply = {"final": None}
        if reply.get("final"):
            on_step({"step": step + 1, "final": True, "thought": reply.get("thought", "")})
            return tidy(reply["final"])
        name, args = reply.get("tool"), reply.get("args") or {}
        if name not in tools:
            scratch += f"\nStep {step + 1}: invalid tool {name!r}. Use a listed tool or give final."
            continue
        try:
            result = tools[name](**args)
            obs = json.dumps(result, default=str)[:6000]
        except Exception as e:
            obs = f"ERROR: {e}"
        on_step({"step": step + 1, "tool": name, "args": args, "thought": reply.get("thought", ""),
                 "observation": obs[:400]})
        scratch += f"\nStep {step + 1}: {name}({json.dumps(args)}) -> {obs}"
    final = router.chat("deep", "Analyst final",
                        SYSTEM.split("Tools:")[0] + "Tools are no longer available. Answer the user's QUESTION now, in "
                        "Markdown, using only the research below. If the research did not find the recall, regulation "
                        "or fact asked about, say clearly that it could not be found and may not exist. Never ask the "
                        "user to repeat the question.",
                        f"QUESTION: {question}\n\nResearch so far:\n{(scratch or 'none')[-12000:]}\n\n"
                        f"Now answer this QUESTION: {question}")
    tool_call = re.match(r"\s*(Step\s*\d+\s*:\s*)?`?(" + "|".join(tools) + r")`?\s*\(", final)
    if final.lstrip().startswith("{") or tool_call:  # model still tried to call a tool
        parsed = parse_json(final, {})
        final = parsed.get("final") if isinstance(parsed, dict) and parsed.get("final") else (
            "I couldn't find reliable information to answer this. The searches returned no matching records.")
    return tidy(final)


def tidy(answer: str) -> str:
    """Turn model-internal citation markers like 【3†L1-L4】 into plain [3]."""
    return re.sub(r"【(\d+)[^】]*】", r" [\1]", re.sub(r"【[^\d】][^】]*】", "", answer or ""))
