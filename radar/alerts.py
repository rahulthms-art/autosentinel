"""Push alerts to Slack / Microsoft Teams / any webhook that accepts {"text": "..."}."""
from __future__ import annotations

import logging

import requests

log = logging.getLogger("radar")


def format_alert(watchlist: str, stats: dict, findings: list[dict], signals: list[dict]) -> str:
    lines = [f"*AutoSentinel: {watchlist}*: {stats['high']} new High-severity items, "
             f"{stats.get('priority_signals', 0)} Priority early-warning signals"]
    for f in [f for f in findings if f.get("severity") == "High"][:5]:
        lines.append(f"• [{f['type']}] {f['title']}: {f.get('action') or ''} {f.get('url') or ''}")
    for s in [s for s in signals if not s["recalled"] and s.get("priority") == "Priority"][:3]:
        lines.append(f"• ⚠️ Priority complaint spike: {s['vehicle']}: {s['component']} (x{s['ratio']}, no recall yet)")
    return "\n".join(lines)


def send(webhook: str, watchlist: str, stats: dict, findings: list[dict], signals: list[dict]) -> bool:
    try:
        requests.post(webhook, json={"text": format_alert(watchlist, stats, findings, signals)},
                      timeout=15).raise_for_status()
        return True
    except requests.RequestException as e:
        log.warning("Alert webhook failed: %s", e)
        return False
