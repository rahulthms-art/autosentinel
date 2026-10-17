"""Runtime settings, read from environment variables (or passed in from the UI)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    nebius_api_key: str = field(default_factory=lambda: os.getenv("NEBIUS_API_KEY", ""))
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    nebius_base_url: str = field(default_factory=lambda: os.getenv(
        "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"))
    fast_model: str = field(default_factory=lambda: os.getenv("NEMOTRON_FAST", ""))
    deep_model: str = field(default_factory=lambda: os.getenv("NEMOTRON_DEEP", ""))
    db_path: str = field(default_factory=lambda: os.getenv(
        "RADAR_DB", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "radar.db")))
    alert_webhook: str = field(default_factory=lambda: os.getenv("ALERT_WEBHOOK_URL", ""))

    def __post_init__(self):
        # Treat template placeholders ("paste-your-...", "your-...-key") as missing
        for name in ("nebius_api_key", "tavily_api_key", "alert_webhook"):
            val = (getattr(self, name) or "").strip()
            if val.startswith(("paste-your", "your-")) or val.endswith("-here"):
                val = ""
            setattr(self, name, val)

    def missing(self) -> list[str]:
        return [name for name, val in [("NEBIUS_API_KEY", self.nebius_api_key),
                                       ("TAVILY_API_KEY", self.tavily_api_key)] if not val]


# Official / authoritative domains searched per market (Tavily include_domains).
MARKET_DOMAINS = {
    "EU": ["ec.europa.eu", "eur-lex.europa.eu", "unece.org", "acea.auto"],
    "UK": ["gov.uk", "legislation.gov.uk", "smmt.co.uk"],
    "NL": ["rdw.nl", "rijksoverheid.nl", "government.nl", "ilent.nl"],
    "US": ["nhtsa.gov", "federalregister.gov", "epa.gov"],
    "Canada": ["tc.canada.ca", "canada.ca"],
    "Japan": ["mlit.go.jp"],
    "Australia": ["infrastructure.gov.au", "productsafety.gov.au"],
}
# Reputable automotive trade press, used for supplier risk and as the fallback when
# official domains return nothing. The open web is never searched unrestricted.
TRADE_PRESS = ["reuters.com", "autonews.com", "automotivenews.com", "just-auto.com", "electrive.com",
               "automotiveworld.com", "sae.org", "autoweek.com", "insideevs.com", "theverge.com"]
SUPPLIER_DOMAINS = TRADE_PRESS
# "Global" queries: regulators across markets + trade press
GLOBAL_DOMAINS = sorted({d for ds in MARKET_DOMAINS.values() for d in ds} | set(TRADE_PRESS))
EXCLUDE_DOMAINS = ["facebook.com", "instagram.com", "tiktok.com", "x.com", "twitter.com",
                   "pinterest.com", "youtube.com", "quora.com"]

TOPICS = ["Recalls", "Regulations", "Supplier risk"]
SEVERITIES = ["High", "Medium", "Low"]
