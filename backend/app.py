"""
FastAPI service for the real-time fraud detection engine.

  REST        /api/*     scoring, ingestion, alerts, configuration, export
  WebSocket   /ws        live push of scored transactions + stats to the dashboard
  Static      /          the dashboard (../frontend)

Run:  uvicorn app:app --host 0.0.0.0 --port 8000

Environment variables (all optional)
  FRAUD_API_KEY      if set, state-changing endpoints require the header  X-API-Key: <value>
  FRAUD_THRESHOLD    initial alert threshold (default from engine.py)
  FRAUD_STREAM_RATE  simulated transactions per second (default 6)
  FRAUD_SIMULATE     "0" disables the built-in simulator, so only /api/ingest traffic is scored
  FRAUD_CORS_ORIGINS comma-separated allowed origins (default "*", fine for a demo only)
  FRAUD_LOG_LEVEL    logging level (default INFO)
"""
from __future__ import annotations

import asyncio
import csv
import hmac
import io
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np
from engine import DEFAULT_THRESHOLD, FraudEngine, public
from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from simulator import ALL_CITIES, CATEGORIES, MERCHANT_RISK, SCENARIOS, Simulator

logging.basicConfig(level=os.getenv("FRAUD_LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("fraudsentinel")

API_KEY = os.getenv("FRAUD_API_KEY") or None
SIMULATE = os.getenv("FRAUD_SIMULATE", "1") != "0"
CORS_ORIGINS = [o.strip() for o in os.getenv("FRAUD_CORS_ORIGINS", "*").split(",") if o.strip()]
TICK = 0.25  # seconds between micro-batches
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

engine = FraudEngine(threshold=float(os.getenv("FRAUD_THRESHOLD", DEFAULT_THRESHOLD)))
sim: Simulator | None = None
cfg = {"rate": float(os.getenv("FRAUD_STREAM_RATE", 6.0)), "paused": not SIMULATE, "fraud_rate": 0.04}
clients: set = set()


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not serialisable: {type(o)}")


def config_view():
    return {**cfg, "threshold": engine.threshold}


async def broadcast(msg):
    if not clients:
        return
    data = json.dumps(msg, default=_default)
    dead = []

    async def _send(ws):
        try:
            await ws.send_text(data)
        except Exception:
            dead.append(ws)

    await asyncio.gather(*(_send(ws) for ws in list(clients)))
    for ws in dead:
        clients.discard(ws)


async def stream_loop():
    accum, idle = 0.0, 0
    while True:
        await asyncio.sleep(TICK)
        try:
            batch = []
            if not cfg["paused"]:
                accum += cfg["rate"] * TICK
                n = int(accum)
                accum -= n
                for _ in range(n):
                    t = sim.next_txn()
                    t["_t_in"] = time.perf_counter()
                    batch.append(t)
            results = engine.process(batch) if batch else []
            if batch:
                engine.clock = max(engine.clock, sim.clock)
            idle += 1
            if results or idle >= 4:
                idle = 0
                await broadcast({"type": "tick", "txns": [public(r) for r in results],
                                 "stats": engine.stats(), "config": config_view()})
        except Exception:  # never let the stream die
            log.exception("stream loop error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global sim
    log.info("Training anomaly model ...")
    engine.train()
    log.info("Model ready: %s", engine.train_info)
    if API_KEY is None:
        log.warning("FRAUD_API_KEY is not set: state-changing endpoints are open (demo mode)")
    sim = Simulator(seed=None, fraud_rate=cfg["fraud_rate"], start_ts=time.time())
    task = asyncio.create_task(stream_loop())
    yield
    task.cancel()


app = FastAPI(title="FraudSentinel - Real-time Fraud Detection Engine", version="1.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


# --------------------------------------------------------------------------- auth
def require_key(x_api_key: str | None = Header(default=None)):
    """No-op in demo mode. With FRAUD_API_KEY set, state-changing endpoints need a matching X-API-Key header."""
    if API_KEY is not None and not (x_api_key and hmac.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(401, "Missing or invalid X-API-Key")


# --------------------------------------------------------------------------- models
class TxnIn(BaseModel):
    user_id: str
    amount: float = Field(gt=0, le=1e9)
    merchant: str = "Unknown"
    category: str = "Groceries"
    city: str = "Mumbai"
    country: str = "IN"
    lat: float | None = None
    lon: float | None = None
    device_id: str = "unknown"
    timestamp: float | None = None
    merchant_risk: float | None = Field(default=None, ge=0, le=1)
    id: str | None = None


class WhatIf(BaseModel):
    user_id: str | None = None
    amount: float = Field(gt=0, le=1e8)
    category: str = "Electronics"
    city: str = "Mumbai"
    new_device: bool = False
    minutes_since_last: float = Field(30, ge=0, le=100000)
    txns_last_minute: int = Field(0, ge=0, le=50)


class ConfigIn(BaseModel):
    threshold: float | None = Field(default=None, ge=0.2, le=0.95)
    rate: float | None = Field(default=None, ge=0.5, le=60)
    fraud_rate: float | None = Field(default=None, ge=0, le=0.3)
    paused: bool | None = None


class ResolveIn(BaseModel):
    status: str = Field(pattern="^(open|confirmed|dismissed)$")


# --------------------------------------------------------------------------- REST
@app.get("/api/health")
def health():
    return {"status": "ok" if engine.model is not None else "starting", "model_ready": engine.model is not None,
            "uptime_s": int(time.time() - engine.started), "clients": len(clients),
            "auth_required": API_KEY is not None, **config_view()}


@app.get("/api/meta")
def meta():
    return {
        "categories": [{"name": k, "risk": v["risk"]} for k, v in CATEGORIES.items()],
        "cities": [{"name": k, "country": v[3]} for k, v in ALL_CITIES.items()],
        "attacks": SCENARIOS,
        "model": engine.train_info,
    }


@app.get("/api/stats")
def get_stats():
    return engine.stats()


@app.get("/api/transactions")
def get_transactions(limit: int = 50, flagged_only: bool = False):
    items = [public(r) for r in reversed(engine.recent) if r["flagged"] or not flagged_only]
    return items[: max(1, min(limit, 300))]


@app.get("/api/alerts")
def get_alerts(status: str | None = None, limit: int = 100):
    items = [public(engine.alerts[i]) for i in reversed(engine.alert_order) if i in engine.alerts]
    if status:
        items = [a for a in items if a["status"] == status]
    return items[: max(1, min(limit, 500))]


@app.post("/api/alerts/{alert_id}/resolve", dependencies=[Depends(require_key)])
async def resolve_alert(alert_id: str, body: ResolveIn):
    a = engine.resolve(alert_id, body.status)
    if a is None:
        raise HTTPException(404, "Alert not found (it may have aged out of memory)")
    await broadcast({"type": "alert_update", "id": alert_id, "status": body.status, "stats": engine.stats()})
    return public(a)


@app.post("/api/ingest", dependencies=[Depends(require_key)])
async def ingest(txn: TxnIn):
    """Production entry point: send a real transaction, get a decision back instantly."""
    t_in = time.perf_counter()
    if txn.category not in CATEGORIES:
        raise HTTPException(422, f"category must be one of {list(CATEGORIES)}")
    lat, lon = txn.lat, txn.lon
    if lat is None or lon is None:
        city = ALL_CITIES.get(txn.city)
        if not city:
            raise HTTPException(422, "Provide lat/lon or a known city name (see /api/meta)")
        lat, lon = city[1], city[2]
    risk = txn.merchant_risk if txn.merchant_risk is not None else MERCHANT_RISK.get(
        txn.merchant, CATEGORIES[txn.category]["risk"])
    t = dict(id=txn.id or f"API-{uuid.uuid4().hex[:12]}", user_id=txn.user_id, user_name=txn.user_id,
             amount=txn.amount, merchant=txn.merchant, category=txn.category, merchant_risk=risk, city=txn.city,
             country=txn.country, lat=lat, lon=lon, device_id=txn.device_id,
             ts=txn.timestamp if txn.timestamp is not None else engine.clock, _t_in=t_in)
    r = engine.process([t])[0]
    await broadcast({"type": "tick", "txns": [public(r)], "stats": engine.stats(), "config": config_view()})
    return public(r)


@app.post("/api/score")
def score(p: WhatIf):
    """Dry-run risk check (used by the 'Risk Checker' panel). Nothing is stored."""
    if p.category not in CATEGORIES:
        raise HTTPException(422, "Unknown category")
    if p.city not in ALL_CITIES:
        raise HTTPException(422, "Unknown city")
    return engine.what_if(p.model_dump())


@app.get("/api/config")
def get_config():
    return config_view()


@app.post("/api/config", dependencies=[Depends(require_key)])
async def set_config(c: ConfigIn):
    if c.threshold is not None:
        engine.threshold = c.threshold
    if c.rate is not None:
        cfg["rate"] = c.rate
    if c.fraud_rate is not None:
        cfg["fraud_rate"] = c.fraud_rate
        sim.fraud_rate = c.fraud_rate
    if c.paused is not None:
        cfg["paused"] = c.paused
    await broadcast({"type": "config", "config": config_view()})
    return config_view()


@app.post("/api/simulate/attack", dependencies=[Depends(require_key)])
def simulate_attack(kind: str = "random"):
    if kind != "random" and kind not in SCENARIOS:
        raise HTTPException(422, f"kind must be 'random' or one of {SCENARIOS}")
    n = sim.inject(None if kind == "random" else kind)
    return {"queued": n, "kind": kind}


@app.post("/api/reset", dependencies=[Depends(require_key)])
async def reset():
    engine.reset()
    await broadcast({"type": "reset", "stats": engine.stats(), "config": config_view()})
    return {"ok": True}


@app.get("/api/export/alerts.csv")
def export_alerts():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "time", "user_id", "amount_inr", "merchant", "category", "city", "country",
                "score", "level", "status", "reasons"])
    for i in engine.alert_order:
        a = engine.alerts.get(i)
        if a:
            w.writerow([a["id"], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a["received_at"])), a["user_id"],
                        a["amount"], a["merchant"], a["category"], a["city"], a["country"], a["score"], a["level"],
                        a["status"], " | ".join(r["title"] for r in a["reasons"])])
    buf.seek(0)
    return StreamingResponse(iter([buf.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=fraud_alerts.csv"})


# --------------------------------------------------------------------------- WebSocket
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        snap = {
            "type": "snapshot",
            "txns": [public(r) for r in list(engine.recent)[-80:]],
            "alerts": [public(engine.alerts[i]) for i in list(engine.alert_order)[-100:] if i in engine.alerts],
            "stats": engine.stats(),
            "config": config_view(),
        }
        await ws.send_text(json.dumps(snap, default=_default))
        while True:
            await ws.receive_text()  # keep-alive; client messages are ignored
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        clients.discard(ws)


if os.path.isdir(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
