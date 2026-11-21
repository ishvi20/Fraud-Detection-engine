import math

import numpy as np
import pytest
from engine import FEATURES, FeatureStore, FraudEngine, apply_rules, haversine, public
from simulator import ALL_CITIES, CATEGORIES, MERCHANT_RISK, Simulator

T0 = 1_700_000_000.0


@pytest.fixture(scope="module")
def trained():
    e = FraudEngine()
    e.train(n=6000, start_ts=T0 - 86400)
    return e


@pytest.fixture()
def engine(trained):
    trained.store = FeatureStore()
    trained.reset()
    return trained


def txn(i, user="U1", amount=500.0, cat="Groceries", city="Mumbai", ts=T0, device="d1"):
    name, lat, lon, cc = ALL_CITIES[city]
    merchant = CATEGORIES[cat]["merchants"][0]
    return dict(id=f"T{i}", user_id=user, user_name=user, amount=amount, merchant=merchant, category=cat,
                merchant_risk=MERCHANT_RISK[merchant], city=name, lat=lat, lon=lon, country=cc, device_id=device, ts=ts)


def warm_up(engine, user="U1", n=8):
    """Give a customer a normal history: small grocery payments in Mumbai, hours apart."""
    for i in range(n):
        engine.process([txn(f"w{i}", user=user, amount=400 + 10 * i, ts=T0 + i * 3600)])
    return T0 + n * 3600


def codes(r):
    return {x["code"] for x in r["reasons"]}


# ---------------------------------------------------------------- helpers
def test_haversine_mumbai_delhi():
    d = haversine(19.0760, 72.8777, 28.6139, 77.2090)
    assert 1100 < d < 1200
    assert haversine(10, 10, 10, 10) == pytest.approx(0, abs=1e-9)


def test_simulator_is_deterministic():
    a = [Simulator(seed=5, start_ts=T0).next_txn() for _ in range(1)]
    s1, s2 = Simulator(seed=5, start_ts=T0), Simulator(seed=5, start_ts=T0)
    seq1 = [(t["user_id"], t["amount"]) for t in (s1.next_txn() for _ in range(200))]
    seq2 = [(t["user_id"], t["amount"]) for t in (s2.next_txn() for _ in range(200))]
    assert seq1 == seq2 and a


def test_online_stats_match_numpy():
    store = FeatureStore()
    amounts = [120.0, 340.0, 95.0, 800.0, 410.0, 60.0, 230.0]
    for i, a in enumerate(amounts):
        store.extract(txn(i, amount=a, ts=T0 + i * 600))
    u = store.users["U1"]
    logs = np.log(amounts)
    assert u.mean == pytest.approx(logs.mean())
    assert u.m2 / (u.n - 1) == pytest.approx(logs.var(ddof=1))


# ---------------------------------------------------------------- rules + scoring
def test_normal_payment_is_not_flagged(engine):
    ts = warm_up(engine)
    r = engine.process([txn("n1", amount=450, ts=ts + 3600)])[0]
    assert not r["flagged"]
    assert r["status"] is None
    assert 0 <= r["score"] <= 1


def test_impossible_travel_is_flagged(engine):
    ts = warm_up(engine)
    engine.process([txn("a", amount=500, ts=ts)])
    r = engine.process([txn("b", amount=60000, cat="Electronics", city="London", ts=ts + 300, device="new")])[0]
    assert "IMPOSSIBLE_TRAVEL" in codes(r)
    assert r["flagged"]
    assert r["level"] in ("high", "critical")


def test_velocity_burst_triggers_velocity_rule(engine):
    ts = warm_up(engine)
    results = [engine.process([txn(f"v{i}", amount=2500, cat="Electronics", ts=ts + i * 4)])[0] for i in range(8)]
    assert "VELOCITY" in codes(results[-1])
    assert results[-1]["flagged"]
    assert not results[0]["flagged"], "the first payment of a burst looks normal; velocity needs history"


def test_card_testing_pattern(engine):
    ts = warm_up(engine)
    rs = [engine.process([txn(f"c{i}", amount=9, cat="Subscriptions", ts=ts + i * 3, device="new")])[0] for i in range(5)]
    assert "CARD_TESTING" in codes(rs[-1])


def test_noisy_rules_are_scaled_down():
    t = txn(1, amount=20000, cat="Crypto Exchange")
    f = {"amount_z": 0.0, "new_device": 0}
    c = dict(distance_km=0, speed_kmh=0, gap_s=3600, typical=500, hour=14, v1=0, v10=0, home_km=0)
    _, reasons = apply_rules(t, f, c)
    risky = next(r for r in reasons if r["code"] == "RISKY_MERCHANT")
    assert risky["weight"] < 0.2  # raw weight is 0.35 for >= 10k; scaled by RULE_SCALE


def test_score_is_probability_like(engine):
    sim = Simulator(seed=3, start_ts=T0, fraud_rate=0.05)
    res = engine.process([sim.next_txn() for _ in range(400)])
    s = np.array([r["score"] for r in res])
    assert s.min() >= 0 and s.max() <= 1
    assert all(len(r["_x"]) == len(FEATURES) for r in res)


def test_ranking_quality_regression_guard(engine):
    """Cheap guard against accidentally breaking the model or the fusion: ROC-AUC must stay high."""
    from sklearn.metrics import roc_auc_score
    sim = Simulator(seed=77, start_ts=T0, fraud_rate=0.05)
    res = []
    for _ in range(300):
        res += engine.process([sim.next_txn() for _ in range(8)])
    y = [bool(r["_truth"]) for r in res]
    assert sum(y) > 30
    assert roc_auc_score(y, [r["score"] for r in res]) > 0.9


# ---------------------------------------------------------------- state + workflow
def test_what_if_does_not_mutate_state(engine):
    warm_up(engine)
    before = (engine.total, len(engine.store.users["U1"].hist), engine.store.users["U1"].n)
    r = engine.what_if(dict(user_id="U1", amount=90000, category="Electronics", city="Dubai", new_device=True,
                            minutes_since_last=5, txns_last_minute=0))
    assert r["id"] == "WHAT-IF" and "_x" not in r
    after = (engine.total, len(engine.store.users["U1"].hist), engine.store.users["U1"].n)
    assert before == after


def test_alert_resolution_updates_counters(engine):
    ts = warm_up(engine)
    engine.process([txn("a", ts=ts)])
    r = engine.process([txn("b", amount=80000, cat="Wire Transfer", city="Lagos", ts=ts + 200, device="new")])[0]
    assert r["flagged"] and engine.status_counts["open"] == 1
    engine.resolve(r["id"], "confirmed")
    s = engine.stats()
    assert s["status_counts"]["open"] == 0 and s["status_counts"]["confirmed"] == 1
    assert s["money_confirmed"] == 80000
    engine.resolve(r["id"], "dismissed")
    assert engine.stats()["money_confirmed"] == 0
    assert engine.resolve("missing", "confirmed") is None


def test_public_strips_private_fields(engine):
    r = engine.process([txn("p")])[0]
    assert "_truth" in r and "_x" in r
    assert not any(k.startswith("_") for k in public(r))


def test_flagged_is_consistent_with_threshold(engine):
    sim = Simulator(seed=9, start_ts=T0, fraud_rate=0.05)
    for thr in (0.5, 0.8):
        engine.threshold = thr
        res = engine.process([sim.next_txn() for _ in range(300)])
        assert all(r["flagged"] == (r["score"] >= thr) for r in res)
        assert math.isfinite(engine.stats()["avg_detection_ms"])
    assert engine.flagged > 0
