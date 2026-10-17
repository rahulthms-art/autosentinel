"""Brand-safe display mode for screen recordings and public demos.

Replaces vehicle makes, model names, branded driver-assist products and suppliers with neutral
aliases ("OEM A", "Vehicle 1", "Supplier S1"), and masks identifiers that could be looked up to
reveal a brand (VINs, NHTSA campaign numbers, links). Display-only: stored data and the AI's
inputs stay real. Hackathon sponsors (NVIDIA, Nebius, Tavily) are not masked.
"""
from __future__ import annotations

import re
import string

# Distinctive names: matched case-insensitively on word boundaries
MAKES = ["Mercedes-Benz", "Land Rover", "Range Rover", "General Motors", "Harley-Davidson", "Alfa Romeo",
         "Aston Martin", "Rolls-Royce", "Li Auto", "Tesla", "Chevrolet", "Chevy", "Hyundai", "Kia", "Volkswagen",
         "Audi", "Porsche", "BMW", "Mercedes", "Volvo", "Polestar", "Ford", "Lincoln", "Rivian", "Lucid", "Nissan",
         "Infiniti", "Toyota", "Lexus", "Honda", "Acura", "Subaru", "Mazda", "Mitsubishi", "Jeep", "Dodge",
         "Chrysler", "Stellantis", "Fiat", "Peugeot", "Citroen", "Citroën", "Opel", "Vauxhall", "Renault", "Dacia",
         "Skoda", "Škoda", "Cupra", "BYD", "NIO", "XPeng", "Geely", "SAIC", "Jaguar", "Cadillac", "Buick", "GMC",
         "Fisker", "Zeekr", "Leapmotor", "VinFast", "Maserati", "Ferrari", "Lamborghini", "Bentley", "Suzuki",
         "Isuzu", "Daimler", "Scania", "Iveco", "GM", "VW"]
# Brand names that are also ordinary words: matched only with exact capitalisation
MAKES_CASED = ["Mini", "Ram", "Genesis", "Lotus", "SEAT", "MG"]
MODELS = ["Mustang Mach-E", "F-150 Lightning", "Bolt EUV", "Bolt EV", "Model 3", "Model S", "Model X", "Model Y",
          "Cybertruck", "Ioniq 5", "Ioniq 6", "Ioniq 9", "IONIQ 5", "IONIQ 6", "Ioniq", "ID. Buzz", "ID.Buzz",
          "ID.3", "ID.4", "ID.5", "ID.7", "EV3", "EV6", "EV9", "Mach-E", "F-150", "F150", "R1T", "R1S", "R2",
          "Ariya", "Kona", "Niro", "Taycan", "iX3", "iX1", "i4", "i5", "i7", "iX", "EQS", "EQE", "EQA", "EQB",
          "EX90", "EX30", "XC40", "XC60", "XC90", "Silverado", "Equinox EV", "Blazer EV", "Hummer EV", "Lyriq",
          "Prius", "bZ4X", "RAV4", "CR-V", "Civic", "Accord", "Grand Cherokee", "Wrangler", "Outback", "Forester",
          "Telluride", "Tucson", "Santa Fe", "Sportage", "Sorento", "e-tron", "Q4 e-tron", "Enyaq", "Megane",
          "Mégane", "Zoe", "Atto 3", "Qashqai", "Juke", "X-Trail", "Rogue", "Altima", "Golf", "Polo", "Tiguan",
          "Camry", "Corolla", "Highlander", "Explorer", "Escape", "Bronco", "Maverick", "Ranger", "Transit",
          "ID4", "Sportback", "Yukon", "Escalade", "Tahoe", "Suburban", "Sierra", "Hummer", "Lyriq",
          "Model 3 Highland", "Cybercab", "Ioniq 5 N", "EV5", "EV4"]
MODELS_CASED = ["Leaf", "Bolt", "Seal", "Dolphin", "Lightning", "Mustang", "Ioniq"]
PRODUCTS = ["Full Self-Driving", "Self-Driving (Supervised)", "FSD", "Autopilot", "Super Cruise", "BlueCruise",
            "HW3", "HW4", "Hardware 4", "Hardware 3",
            "Highway Driving Assist", "Travel Assist", "ProPILOT", "Drive Pilot", "Pilot Assist", "Autosteer"]
SUPPLIERS = ["LG Energy Solution", "Samsung SDI", "SK On", "ZF Friedrichshafen", "Bosch", "Continental",
             "Denso", "CATL", "Panasonic", "Autoliv", "Takata", "Joyson", "Aptiv", "Valeo", "Magna", "Mobileye",
             "Northvolt", "Lear", "Brembo", "ARC Automotive", "Infineon", "NXP", "Renesas", "FICOSA", "Ficosa",
             "LG", "ZF"]

RE_VIN = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
RE_CAMPAIGN = re.compile(r"\b(\d{2})([VEC])\d{6}\b")          # NHTSA campaign numbers, e.g. 24V700000
RE_URL = re.compile(r"https?://\S*")
# Other regulators' public reference numbers (also look-up-able): Transport Canada, EU Safety Gate, UK DVSA
RE_OTHER_REFS = [(re.compile(r"\b20\d{5}\b"), "•••••••"),   # Transport Canada: year + 3 digits
                 (re.compile(r"\bA\d{2}/\d{5}/\d{2}\b"), "A••/•••••/••"),
                 (re.compile(r"\bR/\d{4}/\d{2,4}\b"), "R/••••/•••")]
RE_MD_LINK = re.compile(r"\[([^\]]+)\]\((https?://[^)]+)\)")


def _letters(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = string.ascii_uppercase[r] + s
    return s


class Anonymizer:
    def __init__(self, vehicles: list[dict] | None = None, enabled: bool = True):
        self.enabled = enabled
        vehicles = vehicles or []
        # Watchlist makes get the first letters, in watchlist order
        order = []
        for v in vehicles:
            mk = v.get("make", "").strip()
            if mk and mk.lower() not in [o.lower() for o in order]:
                order.append(mk)
        self.make_alias = {mk.lower(): f"OEM {_letters(i)}" for i, mk in enumerate(order)}
        self._synonyms = {"chevy": "chevrolet", "vw": "volkswagen", "mercedes": "mercedes-benz",
                          "gm": "general motors", "citroën": "citroen", "škoda": "skoda"}
        self.model_alias = {}
        for i, v in enumerate(vehicles, 1):
            if v.get("model"):
                self.model_alias[v["model"].strip().lower()] = f"Vehicle {i}"
        rules: list[tuple[re.Pattern, object]] = []

        def word(name, cased=False):
            flags = 0 if cased else re.I
            return re.compile(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", flags)

        # Longest first so "Bolt EV" wins over "Bolt", "Mercedes-Benz" over "Mercedes"
        models = sorted(set(MODELS) | {v["model"] for v in vehicles if v.get("model")}, key=len, reverse=True)
        for m in models:
            rules.append((word(m), lambda _m, m=m: self.model_alias.get(m.lower(), "a vehicle model")))
        for m in sorted(MODELS_CASED, key=len, reverse=True):
            rules.append((word(m, cased=True), lambda _m, m=m: self.model_alias.get(m.lower(), "a vehicle model")))
        for p in sorted(PRODUCTS, key=len, reverse=True):
            rules.append((word(p), "the driver-assist system"))
        sup = sorted(SUPPLIERS, key=len, reverse=True)
        for i, s in enumerate(sup):
            rules.append((word(s, cased=len(s) <= 3), f"Supplier S{i + 1}"))
        for mk in sorted(MAKES, key=len, reverse=True):
            rules.append((word(mk, cased=len(mk) <= 3), lambda _m, mk=mk: self.make(mk)))
        for mk in MAKES_CASED:
            rules.append((word(mk, cased=True), lambda _m, mk=mk: self.make(mk)))
        self.rules = rules

    def make(self, name: str) -> str:
        """Stable alias: watchlist makes are A, B, C...; others get the next letter on first appearance."""
        key = self._synonyms.get(name.lower(), name.lower())
        if key not in self.make_alias:
            self.make_alias[key] = f"OEM {_letters(len(self.make_alias))}"
        return self.make_alias[key]

    def __call__(self, text):
        if not self.enabled or not isinstance(text, str) or not text:
            return text
        t = RE_MD_LINK.sub(lambda m: m.group(1) + " [link hidden]", text)
        t = RE_URL.sub("[link hidden]", t)
        t = RE_VIN.sub(lambda m: "…" + m.group(0)[-4:], t)
        t = RE_CAMPAIGN.sub(lambda m: f"{m.group(1)}{m.group(2)}••••••", t)
        for pattern, repl in RE_OTHER_REFS:
            t = pattern.sub(repl, t)
        for pattern, repl in self.rules:
            t = pattern.sub(repl if isinstance(repl, str) else repl, t)
        return t

    def vehicle(self, make: str, model: str) -> str:
        if not self.enabled:
            return f"{make} {model}"
        return f"{self.make(make)} {self.model_alias.get(model.lower(), 'vehicle')}"

    def frame(self, df):
        """Mask every text column of a DataFrame."""
        if not self.enabled:
            return df
        out = df.copy()
        for c in out.columns:
            if out[c].dtype == object:
                out[c] = out[c].map(self)
        return out
