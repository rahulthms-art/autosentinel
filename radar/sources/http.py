"""Shared HTTP session with retries and a small TTL cache for public APIs."""
from __future__ import annotations

import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"])))
_session.headers["User-Agent"] = "AutoSentinel/1.0 (+https://github.com)"

_cache: dict = {}
TTL = 3600


def get_json(url: str, params: dict | None = None, timeout: int = 30):
    key = (url, tuple(sorted((params or {}).items())))
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < TTL:
        return hit[1]
    r = _session.get(url, params=params, timeout=timeout)
    if r.status_code == 400 and r.headers.get("content-type", "").startswith("application/json"):
        data = r.json()  # NHTSA answers "no results" with 400 + an empty result list
    else:
        r.raise_for_status()
        data = r.json()
    _cache[key] = (time.time(), data)
    return data


def post(url: str, **kw):
    r = _session.post(url, timeout=kw.pop("timeout", 60), **kw)
    r.raise_for_status()
    return r.json()
