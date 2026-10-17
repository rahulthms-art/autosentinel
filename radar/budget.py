"""Spend control: price every Nemotron call, stop at a budget, and rate-limit the public demo.

Prices are USD per 1M tokens (input, output). Nano is Nebius' published rate as listed by price
aggregators; models without a known price use a deliberately high estimate so the guard errs on
the side of stopping early. Override with PRICE_<SUBSTRING>="in,out", e.g. PRICE_ULTRA="0.6,1.8".
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

DEFAULT_PRICES = {        # model-id substring -> (input, output) USD per 1M tokens
    "nano": (0.06, 0.24),
    "super": (0.40, 1.60),   # estimate
    "ultra": (1.00, 4.00),   # cautious estimate (actual price not published publicly)
}
UNKNOWN_PRICE = (1.00, 4.00)


class BudgetExceeded(RuntimeError):
    pass


class LimitReached(RuntimeError):
    pass


def prices() -> dict:
    out = dict(DEFAULT_PRICES)
    for key, val in os.environ.items():
        if key.startswith("PRICE_"):
            try:
                i, o = (float(x) for x in val.split(","))
                out[key[6:].lower()] = (i, o)
            except ValueError:
                pass
    return out


def price_for(model: str) -> tuple[float, float]:
    m = (model or "").lower()
    for sub, p in prices().items():
        if sub in m:
            return p
    return UNKNOWN_PRICE


def call_cost(model: str, prompt_tokens: int | None, completion_tokens: int | None) -> float:
    pin, pout = price_for(model)
    return ((prompt_tokens or 0) * pin + (completion_tokens or 0) * pout) / 1_000_000


def _env_float(name, default):
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return float(default)


def budget_usd() -> float:
    """Hard stop for this deployment's estimated Nemotron spend."""
    return _env_float("BUDGET_USD", 10)


def demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "").lower() in ("1", "true", "yes")


def limits() -> dict:
    # Sized so that even if a hosted demo restarts daily (losing its spend counter), a fully used day
    # costs about $0.40 at the cautious Ultra price: ~3 x $0.05 runs + ~8 x $0.03 questions.
    return {"runs_per_day": int(_env_float("DEMO_RUNS_PER_DAY", 3)),
            "questions_per_day": int(_env_float("DEMO_QUESTIONS_PER_DAY", 8)),
            "questions_per_session": int(_env_float("DEMO_QUESTIONS_PER_SESSION", 3))}


class Guard:
    """Checked before every model call (Router.before_call) and before each run / question."""

    def __init__(self, db):
        self.db = db

    def spent(self) -> float:
        return sum(call_cost(r["model"], r["prompt_tokens"], r["completion_tokens"])
                   for r in self.db.q("SELECT model, prompt_tokens, completion_tokens FROM llm_calls "
                                      "WHERE step NOT LIKE '[snapshot]%'"))

    def check_budget(self):
        spent, cap = self.spent(), budget_usd()
        if spent >= cap:
            raise BudgetExceeded(f"AI budget reached (${spent:.2f} of ${cap:.2f}). "
                                 "Charts, Fleet check and stored results still work.")

    def today(self, kind: str) -> int:
        day = datetime.now(timezone.utc).date().isoformat()
        return self.db.q("SELECT COUNT(*) AS n FROM usage_events WHERE kind=? AND substr(at,1,10)=?",
                         (kind, day))[0]["n"]

    def allow(self, kind: str, session_count: int = 0):
        """Raise LimitReached if the public demo has used its daily allowance."""
        self.check_budget()
        if not demo_mode():
            return
        lim = limits()
        if kind == "run" and self.today("run") >= lim["runs_per_day"]:
            raise LimitReached(f"The public demo allows {lim['runs_per_day']} radar runs per day. "
                               "Browse the stored results, or try again tomorrow.")
        if kind == "question":
            if session_count >= lim["questions_per_session"]:
                raise LimitReached(f"The public demo allows {lim['questions_per_session']} questions per visit.")
            if self.today("question") >= lim["questions_per_day"]:
                raise LimitReached(f"The public demo allows {lim['questions_per_day']} analyst questions per day.")

    def record(self, kind: str):
        self.db.add_usage(kind)
