"""NVIDIA Nemotron on Nebius Token Factory (OpenAI-compatible API), with tiered routing.

FAST tier (Nemotron Nano/Super): query planning, triage, classification: high volume, cheap.
DEEP tier (Nemotron Ultra/Super): risk briefs, signal analysis, analyst agent: heavy reasoning.
Every call is timed and recorded through `on_call` (the DB audit log).
"""
from __future__ import annotations

import json
import re
import time
from typing import Callable

from openai import OpenAI


def make_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key, max_retries=3, timeout=180)


def list_nvidia_models(client: OpenAI) -> list[str]:
    """NVIDIA model ids available on Token Factory, Nemotron first."""
    ids = [m.id for m in client.models.list().data]
    nv = [i for i in ids if "nvidia" in i.lower() or "nemotron" in i.lower()]
    return sorted(nv, key=lambda i: ("nemotron" not in i.lower(), i))


def pick_default(models: list[str], preferences: list[str]) -> str | None:
    for kw in preferences:
        for m in models:
            if kw in m.lower():
                return m
    return models[0] if models else None


class Router:
    def __init__(self, client: OpenAI, fast_model: str, deep_model: str,
                 on_call: Callable[[dict], None] | None = None,
                 before_call: Callable[[], None] | None = None):
        self.client = client
        self.fast = fast_model
        self.deep = deep_model
        self.on_call = on_call or (lambda rec: None)
        self.before_call = before_call or (lambda: None)  # e.g. budget guard; raises to block the call
        self.trace: list[dict] = []
        # Fast tier runs with Nemotron's reasoning switched off: ~16x fewer tokens for
        # structured planning/triage. Disabled automatically if a model rejects the flag.
        self.fast_no_thinking = True

    def chat(self, tier: str, step: str, system: str, user: str,
             temperature: float = 0.2, max_tokens: int | None = None, thinking: bool | None = None) -> str:
        self.before_call()
        model = self.deep if tier == "deep" else self.fast
        max_tokens = max_tokens or (4096 if tier == "deep" else 2048)
        rec = {"step": step, "model": model, "tier": tier, "ok": False}
        t0 = time.time()
        kwargs = dict(model=model, temperature=temperature, max_tokens=max_tokens,
                      messages=[{"role": "system", "content": system},
                                {"role": "user", "content": user}])
        no_think = (tier == "fast" and self.fast_no_thinking) if thinking is None else not thinking
        rec["thinking"] = not no_think
        try:
            try:
                resp = self.client.chat.completions.create(
                    **kwargs, **({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}} if no_think else {}))
            except Exception as e:
                if not no_think or "400" not in str(e):
                    raise
                self.fast_no_thinking = False  # model doesn't support the switch: retry with defaults
                rec["thinking"] = True
                resp = self.client.chat.completions.create(**kwargs)
            text = strip_thinking(resp.choices[0].message.content or "")
            usage = getattr(resp, "usage", None)
            rec["truncated"] = getattr(resp.choices[0], "finish_reason", None) == "length"
            rec.update(ok=True,
                       prompt_tokens=getattr(usage, "prompt_tokens", None),
                       completion_tokens=getattr(usage, "completion_tokens", None))
            return text
        except Exception as e:
            rec["error"] = str(e)[:500]
            raise
        finally:
            rec["seconds"] = round(time.time() - t0, 2)
            self.trace.append(rec)
            self.on_call(rec)

    def json(self, tier: str, step: str, system: str, user: str, default, max_tokens: int | None = None,
             thinking: bool | None = None):
        text = self.chat(tier, step, system + "\n\nRespond with valid JSON only. No prose, no markdown.", user,
                         temperature=0.0, max_tokens=max_tokens, thinking=thinking)
        return parse_json(text, default)


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def parse_json(text: str, default):
    """Extract the first JSON object/array from a model reply; fall back to `default`."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Several JSON values back to back (e.g. multiple tool calls): take the first valid one
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                return decoder.raw_decode(text, i)[0]
            except json.JSONDecodeError:
                continue
    return default
