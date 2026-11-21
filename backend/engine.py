"""
Hybrid fraud scoring engine.

Pipeline for every transaction
  1. Feature engineering against per-customer state (velocity, geo-distance / implied
     travel speed, spending deviation, merchant risk, device novelty, time of day)
  2. Isolation Forest (unsupervised) -> statistical anomaly score
  3. Explainable business rules -> rule score + human-readable reasons
  4. Noisy-OR fusion -> final risk score in [0, 1]; flagged when >= threshold

All state lives in memory, which is fine for a demo. In production the per-customer
state would sit in Redis / a feature store and the model would be served separately.
"""
from __future__ import annotations

import math
import random
import time
from collections import Counter, deque

import numpy as np
from simulator import ALL_CITIES, CATEGORIES, MERCHANT_RISK, Simulator
from sklearn.ensemble import IsolationForest

FEATURES = [
    "amount_log", "amount_z", "velocity_1m", "velocity_10m", "log_distance", "log_speed",
    "log_home_distance", "hour_sin", "hour_cos", "merchant_risk", "is_foreign", "new_device", "log_gap",
]
FEATURE_LABELS = {
    "amount_log": "Transaction amount",
    "amount_z": "Spend vs. customer history",
    "velocity_1m": "Transactions in last minute",
    "velocity_10m": "Transactions in last 10 min",
    "log_distance": "Distance from last transaction",
    "log_speed": "Implied travel speed",
    "log_home_distance": "Distance from home",
    "hour_sin": "Time of day",
    "hour_cos": "Time of day",
    "merchant_risk": "Merchant risk",
    "is_foreign": "Foreign transaction",
    "new_device": "New device",
    "log_gap": "Time since last transaction",
}
BUCKET = 5  # seconds per timeline bucket
# Fusion settings were chosen on a validation stream (seed 101), never on the test stream used in evaluate.py:
# weights maximise PR-AUC; the threshold maximises F1 (best ~0.69, rounded down to favour recall).
DEFAULT_THRESHOLD = 0.65


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(min(1.0, math.sqrt(a)))


def public(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def _fmt_gap(s):
    if s < 90:
        return f"{s:.0f} sec"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


# --------------------------------------------------------------------------- state
class UserState:
    __slots__ = ("n", "mean", "m2", "hist", "devices", "home")

    def __init__(self, home):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0
        self.hist = deque(maxlen=60)  # (ts, lat, lon)
        self.devices = set()
        self.home = home


class FeatureStore:
    def __init__(self):
        self.users: dict[str, UserState] = {}

    def extract(self, t, update=True, overrides=None):
        u = self.users.get(t["user_id"])
        if u is None:
            u = UserState((t["lat"], t["lon"]))
            if update:
                self.users[t["user_id"]] = u
        ts, lat, lon = t["ts"], t["lat"], t["lon"]
        amt = max(t["amount"], 1.0)
        la = math.log(amt)

        if u.n >= 3:
            std = max(math.sqrt(u.m2 / (u.n - 1)), 0.35)
            z, typical = (la - u.mean) / std, math.exp(u.mean)
        else:
            z, typical = 0.0, amt
        z = max(-5.0, min(15.0, z))

        if u.hist:
            lts, llat, llon = u.hist[-1]
            gap = max(ts - lts, 1.0)
            dist = haversine(llat, llon, lat, lon)
            speed = dist / (gap / 3600.0)
        else:
            gap, dist, speed = 86400.0, 0.0, 0.0

        v1 = sum(1 for h in u.hist if 0 <= ts - h[0] <= 60)
        v10 = sum(1 for h in u.hist if 0 <= ts - h[0] <= 600)
        home_km = haversine(u.home[0], u.home[1], lat, lon)
        new_dev = 1 if (u.n >= 3 and t["device_id"] not in u.devices) else 0
        hour = ((ts / 3600.0) + 5.5) % 24

        c = dict(distance_km=dist, speed_kmh=speed, gap_s=gap, typical=typical, hour=hour, v1=v1, v10=v10, home_km=home_km)
        if overrides and overrides.get("v1"):
            c["v1"] = max(c["v1"], overrides["v1"])
            c["v10"] = max(c["v10"], c["v1"])
        f = {
            "amount_log": math.log1p(amt),
            "amount_z": z,
            "velocity_1m": c["v1"],
            "velocity_10m": c["v10"],
            "log_distance": math.log1p(dist),
            "log_speed": math.log1p(min(speed, 5000.0)),
            "log_home_distance": math.log1p(home_km),
            "hour_sin": math.sin(2 * math.pi * hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour / 24),
            "merchant_risk": t["merchant_risk"],
            "is_foreign": 0 if t["country"] == "IN" else 1,
            "new_device": new_dev,
            "log_gap": math.log1p(min(gap, 86400.0)),
        }

        if update:
            u.hist.append((ts, lat, lon))
            u.n += 1
            d = la - u.mean
            u.mean += d / u.n
            u.m2 += d * (la - u.mean)
            u.devices.add(t["device_id"])
        return f, c


# --------------------------------------------------------------------------- rules
# Evidence scaling per rule. Measured on a labelled simulated stream, VELOCITY / CARD_TESTING / FOREIGN /
# NEW_DEVICE are 88-100% precise, while RISKY_MERCHANT (28%), ODD_HOURS (25%) and GEO_JUMP (4%) mostly fire on
# legitimate payments. Noisy rules are therefore kept as context for the analyst but contribute little score.
RULE_SCALE = {"RISKY_MERCHANT": 0.4, "ODD_HOURS": 0.5, "GEO_JUMP": 0.3}


def apply_rules(t, f, c):
    amt = t["amount"]
    out = []

    def add(code, title, detail, w):
        out.append({"code": code, "title": title, "detail": detail, "weight": round(w * RULE_SCALE.get(code, 1.0), 2)})

    v1, v10 = c["v1"], c["v10"]
    if v1 >= 3:
        add("VELOCITY", "Transaction velocity", f"{v1 + 1} transactions within 60 seconds", min(0.6, 0.30 + 0.08 * (v1 - 3)))
    elif v10 >= 7:
        add("VELOCITY", "Transaction velocity", f"{v10 + 1} transactions within 10 minutes", 0.25)

    if v1 >= 2 and amt <= 60:
        add("CARD_TESTING", "Card-testing pattern", f"Repeated micro-payment of ₹{amt:,.0f}", 0.35)

    d, s = c["distance_km"], c["speed_kmh"]
    if d > 300 and s > 900:
        add("IMPOSSIBLE_TRAVEL", "Impossible travel",
            f"{d:,.0f} km from the previous transaction in {_fmt_gap(c['gap_s'])} (~{s:,.0f} km/h)", 0.65)
    elif d > 300 and s > 350:
        add("GEO_JUMP", "Unusual location jump",
            f"{d:,.0f} km from the previous transaction, {_fmt_gap(c['gap_s'])} earlier", 0.30)

    if f["amount_z"] >= 3.5 and amt >= 8000:
        ratio = amt / max(c["typical"], 1.0)
        w = min(0.60, 0.28 + 0.08 * (f["amount_z"] - 3.5)) + (0.15 if (ratio >= 10 and amt >= 25000) else 0.0)
        add("AMOUNT_SPIKE", "Spending deviation", f"{ratio:,.0f}x this customer's typical spend", min(0.75, w))

    if t["merchant_risk"] >= 0.7:
        add("RISKY_MERCHANT", "High-risk merchant", f"{t['category']} merchant (risk {t['merchant_risk']:.2f})",
            0.20 + (0.15 if amt >= 10000 else 0.0))

    if f["new_device"] and amt >= 8000:
        add("NEW_DEVICE", "Unrecognised device", "First time this device is used for a large payment", 0.30)

    if t["country"] != "IN" and c["home_km"] > 2000:
        add("FOREIGN", "Foreign transaction", f"{t['city']}, {t['country']} - {c['home_km']:,.0f} km from home",
            0.20 + (0.10 if amt >= 20000 else 0.0))

    if 0 <= c["hour"] < 5 and amt >= 15000:
        add("ODD_HOURS", "Unusual hour", "Large payment during the early hours", 0.15)

    return min(1.0, sum(r["weight"] for r in out)), out


# --------------------------------------------------------------------------- engine
class FraudEngine:
    ML_WEIGHT = 1.0
    RULE_WEIGHT = 0.5

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        self.store = FeatureStore()
        self.model = None
        self.threshold = threshold
        self.clock = time.time()
        self.started = time.time()
        self.train_info = {}
        self.reset()

    # ---------------------------------------------------------------- training
    def train(self, n=30000, seed=11, start_ts=None):
        t0 = time.time()
        sim = Simulator(seed=seed, fraud_rate=0.02, start_ts=start_ts or time.time() - 3 * 86400)
        store = FeatureStore()
        rows = []
        for _ in range(n):
            f, _ = store.extract(sim.next_txn())
            rows.append([f[k] for k in FEATURES])
        X = np.array(rows, dtype=float)
        self.model = IsolationForest(n_estimators=120, max_samples=512, contamination="auto",
                                     random_state=seed, n_jobs=1).fit(X)
        raw = -self.model.score_samples(X)
        self.lo, self.hi = float(np.percentile(raw, 50)), float(np.percentile(raw, 99.7))
        self.mu, self.sd = X.mean(axis=0), X.std(axis=0) + 1e-9
        self.train_info = {
            "algorithm": "IsolationForest + rule engine (noisy-OR fusion)",
            "trees": 120, "training_rows": n, "features": len(FEATURES),
            "trained_in_s": round(time.time() - t0, 2),
        }

    # ------------------------------------------------------------------- state
    def reset(self):
        self.total = self.flagged = 0
        self.money_total = 0.0
        self.det_times = deque(maxlen=2000)
        self.det_sum, self.det_n = 0.0, 0
        self.buckets: dict[int, list] = {}
        self.score_hist = [0] * 20
        self.confusion = dict(tp=0, fp=0, fn=0, tn=0)
        self.recent = deque(maxlen=300)
        self.alerts: dict[str, dict] = {}
        self.alert_order = deque(maxlen=500)
        self.amt_by_status = {"open": 0.0, "confirmed": 0.0, "dismissed": 0.0}
        self.status_counts = Counter({"open": 0, "confirmed": 0, "dismissed": 0})
        self.severity, self.reason_counts = Counter(), Counter()
        self.cat_flags, self.country_flags = Counter(), Counter()
        self.merchant_flags: dict[str, list] = {}
        self.user_flags: dict[str, list] = {}

    # ----------------------------------------------------------------- scoring
    def _level(self, score):
        thr = self.threshold
        if score < thr * 0.5:
            return "low"
        if score < thr:
            return "medium"
        return "high" if score < thr + (1 - thr) * 0.5 else "critical"

    def _decide(self, t, f, c, ml, zrow):
        rule, reasons = apply_rules(t, f, c)
        if ml >= 0.5:
            reasons.append({"code": "ML_ANOMALY", "title": "Statistical outlier",
                            "detail": f"Isolation Forest anomaly score {ml:.2f}", "weight": round(self.ML_WEIGHT * ml, 2)})
        final = 1 - (1 - self.ML_WEIGHT * ml) * (1 - self.RULE_WEIGHT * rule)
        flagged = final >= self.threshold
        top = []
        if flagged:
            for i in np.argsort(-np.abs(zrow))[:3]:
                top.append({"feature": FEATURE_LABELS[FEATURES[i]], "z": round(float(zrow[i]), 1),
                            "value": round(float(f[FEATURES[i]]), 2)})
        return {
            "id": t["id"], "ts": t["ts"], "received_at": time.time(),
            "user_id": t["user_id"], "user_name": t.get("user_name", t["user_id"]),
            "amount": t["amount"], "currency": "INR", "merchant": t["merchant"], "category": t["category"],
            "city": t["city"], "country": t["country"], "lat": t["lat"], "lon": t["lon"], "device_id": t["device_id"],
            "score": round(float(final), 3), "ml_score": round(float(ml), 3), "rule_score": round(float(rule), 3),
            "flagged": bool(flagged), "level": self._level(final), "reasons": reasons, "features": top,
            "status": "open" if flagged else None,
            "_truth": t.get("truth"), "_scenario": t.get("scenario"),
        }

    def _score_batch(self, txns, update=True, overrides=None):
        feats, ctxs = [], []
        for t in txns:
            f, c = self.store.extract(t, update=update, overrides=overrides)
            feats.append(f)
            ctxs.append(c)
        X = np.array([[f[k] for k in FEATURES] for f in feats], dtype=float)
        raw = -self.model.score_samples(X)
        ml = np.clip((raw - self.lo) / (self.hi - self.lo), 0, 1)
        Z = (X - self.mu) / self.sd
        out = [self._decide(t, f, c, float(m), z) for t, f, c, m, z in zip(txns, feats, ctxs, ml, Z)]
        for r, x in zip(out, X):
            r["_x"] = x  # raw feature vector; private (stripped by public()), used by offline evaluation
        return out

    def process(self, txns):
        """Score a micro-batch, update state + stats, return full results."""
        start = time.perf_counter()
        results = self._score_batch(txns)
        end = time.perf_counter()
        for t, r in zip(txns, results):
            r["detection_ms"] = round((end - t.get("_t_in", start)) * 1000, 2)
            self._record(r)
        return results

    def what_if(self, p):
        """Dry-run scoring: nothing is stored and no stats are touched."""
        uid = p.get("user_id") or random.choice(list(self.store.users))
        u = self.store.users.get(uid)
        name, lat, lon, cc = ALL_CITIES[p["city"]]
        merchant = CATEGORIES[p["category"]]["merchants"][0]
        last_ts = u.hist[-1][0] if (u and u.hist) else self.clock
        if p.get("new_device") or not (u and u.devices):
            device = "DEV-whatif-new"
        else:
            device = next(iter(u.devices))
        t = dict(id="WHAT-IF", user_id=uid, user_name=uid, amount=p["amount"], merchant=merchant,
                 category=p["category"], merchant_risk=MERCHANT_RISK[merchant], city=name, lat=lat, lon=lon,
                 country=cc, device_id=device, ts=last_ts + p.get("minutes_since_last", 30) * 60)
        start = time.perf_counter()
        r = self._score_batch([t], update=False, overrides={"v1": p.get("txns_last_minute", 0)})[0]
        r["detection_ms"] = round((time.perf_counter() - start) * 1000, 2)
        r["user_id"] = uid
        return public(r)

    # ------------------------------------------------------------------ stats
    def _record(self, r):
        amt = r["amount"]
        self.total += 1
        self.money_total += amt
        self.det_times.append(r["detection_ms"])
        self.det_sum += r["detection_ms"]
        self.det_n += 1
        key = int(r["received_at"] // BUCKET)
        b = self.buckets.setdefault(key, [0, 0])
        b[0] += 1
        if len(self.buckets) > 220:
            for k in [k for k in self.buckets if k < key - 150]:
                del self.buckets[k]
        self.score_hist[min(19, int(r["score"] * 20))] += 1

        truth = r["_truth"]
        if truth is not None:
            k = ("tp" if truth else "fp") if r["flagged"] else ("fn" if truth else "tn")
            self.confusion[k] += 1

        self.recent.append(r)
        if not r["flagged"]:
            return
        b[1] += 1
        self.flagged += 1
        self.severity[r["level"]] += 1
        self.cat_flags[r["category"]] += 1
        self.country_flags[r["country"]] += 1
        for rs in r["reasons"]:
            self.reason_counts[rs["code"]] += 1
        m = self.merchant_flags.setdefault(r["merchant"], [0, 0.0])
        m[0] += 1
        m[1] += amt
        us = self.user_flags.setdefault(r["user_id"], [0, 0.0, r["user_name"]])
        us[0] += 1
        us[1] += amt
        if len(self.alert_order) == self.alert_order.maxlen:
            self.alerts.pop(self.alert_order[0], None)
        self.alert_order.append(r["id"])
        self.alerts[r["id"]] = r
        self.status_counts["open"] += 1
        self.amt_by_status["open"] += amt

    def resolve(self, alert_id, status):
        a = self.alerts.get(alert_id)
        if a is None:
            return None
        old = a["status"]
        if old != status:
            self.status_counts[old] -= 1
            self.status_counts[status] += 1
            self.amt_by_status[old] -= a["amount"]
            self.amt_by_status[status] += a["amount"]
            a["status"] = status
        return a

    def stats(self):
        now_key = int(time.time() // BUCKET)
        keys = range(now_key - 60, now_key)  # completed buckets only; the current one is still filling
        tot = [self.buckets.get(k, [0, 0])[0] for k in keys]
        fl = [self.buckets.get(k, [0, 0])[1] for k in keys]
        tps = sum(self.buckets.get(k, [0, 0])[0] for k in range(now_key - 3, now_key)) / (3 * BUCKET)
        cf = self.confusion
        prec = cf["tp"] / (cf["tp"] + cf["fp"]) if (cf["tp"] + cf["fp"]) else None
        rec = cf["tp"] / (cf["tp"] + cf["fn"]) if (cf["tp"] + cf["fn"]) else None
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else None
        top_m = sorted(self.merchant_flags.items(), key=lambda kv: -kv[1][0])[:5]
        top_u = sorted(self.user_flags.items(), key=lambda kv: -kv[1][1])[:5]
        return {
            "total": self.total,
            "flagged": self.flagged,
            "fraud_rate": round(100 * self.flagged / self.total, 2) if self.total else 0.0,
            "money_total": round(self.money_total, 2),
            "money_at_risk": round(self.amt_by_status["open"] + self.amt_by_status["confirmed"], 2),
            "money_confirmed": round(self.amt_by_status["confirmed"], 2),
            "money_dismissed": round(self.amt_by_status["dismissed"], 2),
            "avg_detection_ms": round(self.det_sum / self.det_n, 2) if self.det_n else 0.0,
            "p95_detection_ms": round(float(np.percentile(self.det_times, 95)), 2) if self.det_times else 0.0,
            "tps": round(tps, 1),
            "uptime_s": int(time.time() - self.started),
            "status_counts": dict(self.status_counts),
            "severity": dict(self.severity),
            "reasons": dict(self.reason_counts),
            "categories": dict(self.cat_flags.most_common(8)),
            "countries": dict(self.country_flags),
            "top_merchants": [{"merchant": k, "count": v[0], "amount": round(v[1], 2)} for k, v in top_m],
            "top_users": [{"user_id": k, "name": v[2], "count": v[0], "amount": round(v[1], 2)} for k, v in top_u],
            "timeline": {"end": (now_key - 1) * BUCKET, "step": BUCKET, "total": tot, "flagged": fl},
            "score_hist": self.score_hist,
            "performance": {"precision": prec, "recall": rec, "f1": f1, **cf},
            "threshold": self.threshold,
        }
