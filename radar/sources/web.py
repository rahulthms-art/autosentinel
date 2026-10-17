"""Tavily web research, restricted to authoritative domains per market where possible."""
from __future__ import annotations

from .http import post

TAVILY_URL = "https://api.tavily.com/search"


def search(api_key: str, query: str, max_results: int = 5, time_range: str | None = None,
           include_domains: list[str] | None = None, topic: str = "general",
           exclude_domains: list[str] | None = None) -> list[dict]:
    payload = {"query": query, "max_results": max_results, "search_depth": "advanced", "topic": topic}
    if time_range:
        payload["time_range"] = time_range
    if include_domains:
        payload["include_domains"] = include_domains
    if exclude_domains:
        payload["exclude_domains"] = exclude_domains
    data = post(TAVILY_URL, json=payload, headers={"Authorization": f"Bearer {api_key}"})
    return [{"title": r.get("title") or r.get("url"), "url": r.get("url"),
             "content": (r.get("content") or "")[:1500], "published": r.get("published_date"),
             "query": query}
            for r in data.get("results", []) if r.get("url")]
