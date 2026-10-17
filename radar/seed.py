"""Default watchlist created on first start, so the demo works out of the box."""
from __future__ import annotations

from .db import DB

DEFAULT = {
    "role": "Compliance & quality",
    "vehicles": [
        {"make": "Tesla", "model": "Model Y", "years": [2021, 2022, 2023, 2024, 2025]},
        {"make": "Chevrolet", "model": "Bolt EV", "years": [2019, 2020, 2021, 2022, 2023]},
        {"make": "Hyundai", "model": "Ioniq 5", "years": [2022, 2023, 2024, 2025]},
        {"make": "Volkswagen", "model": "ID.4", "years": [2021, 2022, 2023, 2024, 2025]},
    ],
    "markets": ["EU", "UK", "US"],
    "components": "HV battery, airbags, brakes, ADAS cameras & radar, infotainment / OTA software",
    "suppliers": "battery cell suppliers, airbag inflator suppliers, semiconductor suppliers",
    "topics": ["Recalls", "Regulations", "Supplier risk", "Early warning"],
    "time_range": "month",
}


def ensure_default(db: DB) -> None:
    if not db.watchlists():
        db.save_watchlist("EV segment watch (EU/UK/US)", DEFAULT)
