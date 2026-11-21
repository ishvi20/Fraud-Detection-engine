"""
Synthetic real-time transaction stream.

This stands in for a real payment-gateway / Kafka feed so the whole system can be
demoed end-to-end without any private data. It models:

  * 500 customers with a home city, personal spending scale, favourite categories
    and 1-3 trusted devices
  * realistic day/night activity and rare relocations between Indian cities
  * five injected fraud scenarios (velocity burst, impossible travel, amount spike,
    account takeover, card testing)

The simulator keeps a *virtual clock* (about 20 s of simulated time per transaction)
so that per-customer features such as "distance since last transaction" and
"transactions in the last minute" behave like they would in production.

Every transaction carries a hidden `truth` label which is only used to compute
precision / recall for the dashboard. The model never sees it.
"""
from __future__ import annotations

import math
import random
import time
import zlib
from collections import deque

CITIES_IN = [
    ("Mumbai", 19.0760, 72.8777, "IN"),
    ("Delhi", 28.6139, 77.2090, "IN"),
    ("Bengaluru", 12.9716, 77.5946, "IN"),
    ("Hyderabad", 17.3850, 78.4867, "IN"),
    ("Chennai", 13.0827, 80.2707, "IN"),
    ("Kolkata", 22.5726, 88.3639, "IN"),
    ("Pune", 18.5204, 73.8567, "IN"),
    ("Ahmedabad", 23.0225, 72.5714, "IN"),
    ("Jaipur", 26.9124, 75.7873, "IN"),
    ("Lucknow", 26.8467, 80.9462, "IN"),
    ("Agra", 27.1767, 78.0081, "IN"),
    ("Chandigarh", 30.7333, 76.7794, "IN"),
]
CITIES_ABROAD = [
    ("Dubai", 25.2048, 55.2708, "AE"),
    ("Singapore", 1.3521, 103.8198, "SG"),
    ("London", 51.5072, -0.1276, "GB"),
    ("New York", 40.7128, -74.0060, "US"),
    ("Lagos", 6.5244, 3.3792, "NG"),
    ("Bangkok", 13.7563, 100.5018, "TH"),
    ("Sao Paulo", -23.5505, -46.6333, "BR"),
]
ALL_CITIES = {c[0]: c for c in CITIES_IN + CITIES_ABROAD}

# risk = baseline merchant-category risk (0..1), mu/sigma = lognormal amount params (INR)
CATEGORIES = {
    "Groceries": dict(risk=0.05, mu=6.0, sigma=0.55, weight=22, merchants=["FreshBasket", "DailyMart", "GreenCart", "Nature's Pantry"]),
    "Restaurants": dict(risk=0.10, mu=6.3, sigma=0.60, weight=18, merchants=["Spice Route", "Urban Tadka", "Cafe Monsoon", "Biryani Bay"]),
    "Fuel": dict(risk=0.10, mu=7.0, sigma=0.40, weight=8, merchants=["PetroPoint", "HighwayFuel", "QuickFill"]),
    "Pharmacy": dict(risk=0.08, mu=5.8, sigma=0.70, weight=6, merchants=["HealthPlus Pharmacy", "MediCare Chemist", "WellNest"]),
    "Utilities": dict(risk=0.05, mu=7.2, sigma=0.50, weight=6, merchants=["PowerGrid BillPay", "AquaFlow Water", "BroadbandOne"]),
    "Subscriptions": dict(risk=0.15, mu=5.9, sigma=0.40, weight=8, merchants=["StreamBox", "MusicWave", "CloudDrive"]),
    "Fashion": dict(risk=0.25, mu=7.6, sigma=0.70, weight=10, merchants=["StyleHub", "Threadly", "UrbanWear"]),
    "Electronics": dict(risk=0.35, mu=9.2, sigma=0.90, weight=5, merchants=["GadgetGalaxy", "VoltMart", "ByteBazaar"]),
    "Travel": dict(risk=0.40, mu=8.6, sigma=0.80, weight=5, merchants=["SkyJet Airlines", "RailYatra", "StayEasy Hotels"]),
    "Online Gaming": dict(risk=0.70, mu=6.5, sigma=0.90, weight=1.5, merchants=["BetKing Arena", "LuckyDraw Games", "PokerNation"]),
    "Crypto Exchange": dict(risk=0.85, mu=8.8, sigma=1.00, weight=0.5, merchants=["CoinVault", "BitBazaar", "ChainSwap"]),
    "Gift Cards": dict(risk=0.80, mu=7.8, sigma=0.80, weight=0.8, merchants=["GiftNation", "VoucherHub"]),
    "Wire Transfer": dict(risk=0.75, mu=9.0, sigma=0.90, weight=0.7, merchants=["QuickRemit", "GlobalSend"]),
    "Jewelry": dict(risk=0.60, mu=9.6, sigma=0.80, weight=0.8, merchants=["Royal Gems", "GoldCrest"]),
}


def _merchant_risk(name: str, base: float) -> float:
    off = (zlib.crc32(name.encode()) % 1000) / 1000.0  # stable per merchant
    return round(min(1.0, max(0.0, base + (off - 0.5) * 0.10)), 3)


MERCHANT_RISK = {m: _merchant_risk(m, c["risk"]) for c in CATEGORIES.values() for m in c["merchants"]}

FIRST = ["Aarav", "Diya", "Vihaan", "Ananya", "Arjun", "Isha", "Kabir", "Meera", "Rohan", "Saanvi", "Aditya", "Priya", "Karan", "Neha"]
LAST = ["Sharma", "Verma", "Iyer", "Patel", "Reddy", "Nair", "Gupta", "Singh", "Das", "Khan", "Mehta", "Joshi"]

SCENARIOS = ["velocity", "geo", "spike", "takeover", "card_testing"]
VELOCITY_CATS = ["Electronics", "Gift Cards", "Fashion", "Online Gaming"]
SPIKE_CATS = ["Electronics", "Jewelry", "Travel", "Fashion"]
TAKEOVER_CATS = ["Crypto Exchange", "Wire Transfer", "Gift Cards"]
GEO_CATS = ["Electronics", "Jewelry", "Gift Cards", "Travel"]


class Simulator:
    def __init__(self, seed=None, n_users=500, fraud_rate=0.04, start_ts=None):
        self.rng = random.Random(seed)
        self.fraud_rate = fraud_rate
        self.clock = start_ts if start_ts is not None else time.time()
        self.seq = 0
        self.pending: deque = deque()
        self.cat_names = list(CATEGORIES)
        self.users = [self._make_user(i) for i in range(n_users)]

    # ------------------------------------------------------------------ users
    def _make_user(self, i):
        rng = self.rng
        city = rng.choice(CITIES_IN)
        return {
            "id": f"U{1000 + i}",
            "name": f"{rng.choice(FIRST)} {rng.choice(LAST)[0]}.",
            "city": city,
            "scale": math.exp(rng.gauss(0, 0.35)),
            "devices": [f"DEV-{rng.getrandbits(32):08x}" for _ in range(rng.choice([1, 2, 2, 3]))],
            "weights": [CATEGORIES[c]["weight"] * rng.uniform(0.3, 2.0) for c in self.cat_names],
            "last_ts": None,
        }

    def _make(self, u, amount, cat, ts, city, device, truth, scenario):
        rng = self.rng
        merchant = rng.choice(CATEGORIES[cat]["merchants"])
        name, lat, lon, cc = city
        self.seq += 1
        u["last_ts"] = max(u["last_ts"] or 0, ts)
        return {
            "id": f"TXN-{self.seq:07d}",
            "user_id": u["id"],
            "user_name": u["name"],
            "amount": round(max(1.0, amount), 2),
            "currency": "INR",
            "merchant": merchant,
            "category": cat,
            "merchant_risk": MERCHANT_RISK[merchant],
            "city": name,
            "country": cc,
            "lat": round(lat + rng.gauss(0, 0.04), 4),
            "lon": round(lon + rng.gauss(0, 0.04), 4),
            "device_id": device,
            "ts": ts,
            "truth": truth,
            "scenario": scenario,
        }

    def _typical(self, u, cat):
        return math.exp(CATEGORIES[cat]["mu"]) * u["scale"]

    # ------------------------------------------------------------ normal flow
    def _normal(self):
        rng = self.rng
        while True:
            self.clock += rng.expovariate(1 / 20.0)
            hour = ((self.clock / 3600.0) + 5.5) % 24  # IST
            if hour < 6 and rng.random() < 0.7:  # quieter at night
                continue
            break
        u = rng.choice(self.users)
        if rng.random() < 0.004 and (u["last_ts"] is None or self.clock - u["last_ts"] > 4 * 3600):
            u["city"] = rng.choice(CITIES_IN)  # moved / travelled
        cat = rng.choices(self.cat_names, weights=u["weights"])[0]
        c = CATEGORIES[cat]
        amount = math.exp(rng.gauss(c["mu"], c["sigma"])) * u["scale"]
        return self._make(u, amount, cat, self.clock, u["city"], rng.choice(u["devices"]), False, None)

    # ------------------------------------------------------------ fraud flows
    def inject(self, kind=None):
        """Queue a fraud scenario. Returns the number of transactions queued."""
        rng = self.rng
        kind = kind if kind in SCENARIOS else rng.choice(SCENARIOS)
        u = rng.choice(self.users)
        t0 = self.clock
        known = rng.choice(u["devices"])
        new_dev = f"DEV-{rng.getrandbits(32):08x}"
        specs = []

        def add(offset, amount, cat, city, device, truth=True):
            specs.append(dict(ts=t0 + offset, u=u, amount=amount, cat=cat, city=city, device=device, truth=truth, scenario=kind))

        if kind == "velocity":
            cat, off = rng.choice(VELOCITY_CATS), 0.0
            for _ in range(rng.randint(6, 10)):
                add(off, rng.uniform(800, 4500), cat, u["city"], known)
                off += rng.uniform(2, 7)

        elif kind == "geo":
            warm = rng.choice(["Groceries", "Restaurants", "Fuel"])
            add(0.0, self._typical(u, warm), warm, u["city"], known, truth=False)
            city = rng.choice(CITIES_ABROAD)
            off = rng.uniform(120, 600)
            dev = new_dev if rng.random() < 0.5 else known
            add(off, rng.uniform(18000, 90000), rng.choice(GEO_CATS), city, dev)
            if rng.random() < 0.6:
                add(off + rng.uniform(40, 120), rng.uniform(8000, 60000), rng.choice(GEO_CATS), city, dev)

        elif kind == "spike":
            cat = rng.choice(SPIKE_CATS)
            amount = max(30000, self._typical(u, cat) * rng.uniform(12, 40))
            add(rng.uniform(10, 60), amount, cat, u["city"], known)

        elif kind == "takeover":
            city = u["city"] if rng.random() < 0.5 else rng.choice(CITIES_ABROAD)
            off = rng.uniform(10, 60)
            for _ in range(rng.randint(1, 2)):
                add(off, rng.uniform(25000, 150000), rng.choice(TAKEOVER_CATS), city, new_dev)
                off += rng.uniform(20, 90)

        elif kind == "card_testing":
            off = 0.0
            for _ in range(rng.randint(4, 6)):
                add(off, rng.uniform(1, 40), "Subscriptions", u["city"], new_dev)
                off += rng.uniform(2, 5)
            add(off + rng.uniform(3, 10), rng.uniform(20000, 60000), "Electronics", u["city"], new_dev)

        self.pending.extend(specs)
        return len(specs)

    def _emit(self, s):
        return self._make(s["u"], s["amount"], s["cat"], s["ts"], s["city"], s["device"], s["truth"], s["scenario"])

    # ---------------------------------------------------------------- stream
    def next_txn(self):
        if not self.pending and self.rng.random() < self.fraud_rate / 3.6:
            self.inject()
        if self.pending and self.rng.random() < 0.45:  # interleave with normal traffic
            return self._emit(self.pending.popleft())
        return self._normal()
