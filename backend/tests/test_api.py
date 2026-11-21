import app as app_module
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with TestClient(app_module.app) as c:  # runs lifespan: trains the model once
        yield c


@pytest.fixture(autouse=True)
def clean(client):
    client.post("/api/reset")
    app_module.engine.threshold = 0.65
    app_module.API_KEY = None
    yield
    app_module.API_KEY = None


def payload(**kw):
    base = dict(user_id="U-test", amount=450, merchant="FreshBasket", category="Groceries", city="Mumbai",
                country="IN", device_id="dev-1", timestamp=1_700_000_000)
    return {**base, **kw}


def test_health_and_meta(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["model_ready"] is True
    m = client.get("/api/meta").json()
    assert {"categories", "cities", "attacks", "model"} <= m.keys()


def test_ingest_scores_and_returns_decision(client):
    r = client.post("/api/ingest", json=payload())
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["score"] <= 1 and "flagged" in body and body["detection_ms"] >= 0
    assert not any(k.startswith("_") for k in body)
    assert client.get("/api/stats").json()["total"] == 1


@pytest.mark.parametrize("bad", [
    dict(amount=-5), dict(amount=0), dict(category="Nope"), dict(city="Atlantis"), dict(merchant_risk=2),
])
def test_ingest_validation(client, bad):
    assert client.post("/api/ingest", json=payload(**bad)).status_code == 422


def test_suspicious_sequence_creates_alert_and_resolves(client):
    for i in range(6):
        client.post("/api/ingest", json=payload(user_id="U-seq", timestamp=1_700_000_000 + i * 3600, amount=400 + i))
    r = client.post("/api/ingest", json=payload(user_id="U-seq", timestamp=1_700_000_000 + 6 * 3600 + 120, amount=95000,
                                                category="Wire Transfer", merchant="QuickRemit", city="Lagos",
                                                country="NG", device_id="stolen-device")).json()
    assert r["flagged"], r
    alerts = client.get("/api/alerts?status=open").json()
    assert [a["id"] for a in alerts] == [r["id"]]
    res = client.post(f"/api/alerts/{r['id']}/resolve", json={"status": "confirmed"})
    assert res.status_code == 200 and res.json()["status"] == "confirmed"
    assert client.get("/api/alerts?status=open").json() == []
    assert client.get("/api/stats").json()["money_confirmed"] == 95000
    csv = client.get("/api/export/alerts.csv")
    assert csv.headers["content-type"].startswith("text/csv") and r["id"] in csv.text


def test_resolve_errors(client):
    assert client.post("/api/alerts/NOPE/resolve", json={"status": "confirmed"}).status_code == 404
    assert client.post("/api/alerts/NOPE/resolve", json={"status": "bogus"}).status_code == 422


def test_score_is_a_dry_run(client):
    client.post("/api/ingest", json=payload())
    before = client.get("/api/stats").json()["total"]
    r = client.post("/api/score", json=dict(user_id="U-test", amount=70000, category="Electronics", city="Dubai",
                                            new_device=True, minutes_since_last=3))
    assert r.status_code == 200 and r.json()["id"] == "WHAT-IF"
    assert client.get("/api/stats").json()["total"] == before


def test_config_bounds_and_update(client):
    assert client.post("/api/config", json={"threshold": 5}).status_code == 422
    assert client.post("/api/config", json={"threshold": 0.8}).json()["threshold"] == 0.8
    assert client.get("/api/config").json()["threshold"] == 0.8


def test_attack_injection(client):
    assert client.post("/api/simulate/attack?kind=velocity").json()["queued"] >= 4
    assert client.post("/api/simulate/attack?kind=nonsense").status_code == 422


def test_api_key_protects_state_changing_endpoints(client):
    app_module.API_KEY = "s3cret"
    assert client.post("/api/ingest", json=payload()).status_code == 401
    assert client.post("/api/reset").status_code == 401
    assert client.post("/api/config", json={"paused": False}).status_code == 401
    assert client.post("/api/ingest", json=payload(), headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/api/ingest", json=payload(), headers={"X-API-Key": "s3cret"}).status_code == 200
    assert client.get("/api/stats").status_code == 200  # reads stay open
    assert client.get("/api/health").json()["auth_required"] is True


def test_websocket_snapshot_and_push(client):
    with client.websocket_connect("/ws") as ws:
        snap = ws.receive_json()
        assert snap["type"] == "snapshot" and "stats" in snap and "config" in snap
        client.post("/api/ingest", json=payload(amount=999))
        msg = ws.receive_json()
        assert msg["type"] == "tick" and msg["txns"][0]["amount"] == 999


def test_ingest_ids_are_unique(client):
    ids = {client.post("/api/ingest", json=payload(user_id=f"U-id{i}")).json()["id"] for i in range(30)}
    assert len(ids) == 30
