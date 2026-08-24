"""
BlueByte AI — Alerts REST API Routes
Provides endpoints for querying, creating, and managing environmental anomaly alerts.

FIXED: previously never imported or called anything from db/queries.py —
every endpoint only ever read from an in-memory list plus 5 hardcoded
SAMPLE_ALERTS, so alerts loaded into Postgres (via the synthetic dataset,
or real ingestion flagging outliers) never appeared here. Now queries
the database first, merges with live (ZMQ-pushed) alerts, and only
falls back to SAMPLE_ALERTS if the database is genuinely unreachable —
same resilience pattern already used in ocean.py/predictions.py.
"""
import sys
import os
import logging
import time
from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Query
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from db import queries

logger = logging.getLogger("API-Alerts")
router = APIRouter()

_live_alerts: list[dict] = []
MAX_LIVE_ALERTS = 100


class AlertCreate(BaseModel):
    alert_type: str
    severity: str
    sensor_id: str
    lat: float
    lon: float
    message: str


SAMPLE_ALERTS = [
    {
        "id": 1, "alert_type": "MARINE_HEATWAVE", "severity": "critical",
        "sensor_id": "INCOIS-ARB-01", "lat": 15.29, "lon": 72.88,
        "message": "Rapid SST Spike (+4.2°C) detected — Potential Coral Bleaching Alert in Arabian Sea",
        "created_at": 1771598000, "acknowledged": False,
    },
    {
        "id": 2, "alert_type": "HYPOXIA_DEAD_ZONE", "severity": "high",
        "sensor_id": "INCOIS-BOB-01", "lat": 13.08, "lon": 82.27,
        "message": "Severe Hypoxia (DO=1.2 mg/L) — Fish Kill / Biomass Mortality Risk in Bay of Bengal",
        "created_at": 1771599000, "acknowledged": False,
    },
    {
        "id": 3, "alert_type": "ALGAL_BLOOM_BURST", "severity": "high",
        "sensor_id": "INCOIS-LAK-01", "lat": 10.56, "lon": 72.64,
        "message": "Abnormal Chlorophyll Surge (Chl-a=8.7 mg/m³) — Harmful Algal Bloom (HAB) near Lakshadweep",
        "created_at": 1771599500, "acknowledged": False,
    },
    {
        "id": 4, "alert_type": "ROUGH_SEA_SURGE", "severity": "medium",
        "sensor_id": "INCOIS-AND-01", "lat": 11.62, "lon": 92.72,
        "message": "Dangerous Wave Surge (Height=5.2m) — Small Craft Fishing Warning near Andaman",
        "created_at": 1771600000, "acknowledged": True,
    },
    {
        "id": 5, "alert_type": "IUU_VESSEL_DETECTED", "severity": "critical",
        "sensor_id": "UNK-IUU-9912", "lat": 10.12, "lon": 74.50,
        "message": "Unregistered vessel operating in Indian EEZ with AIS spoofing risk — IUU alert triggered",
        "created_at": 1771600500, "acknowledged": False,
    },
]


def _normalize_db_alert(row: dict) -> dict:
    """DB rows have created_at as a datetime; sample/live alerts use
    epoch seconds. Normalize to epoch seconds so sorting/merging both
    sources works consistently, and stringify the UUID id."""
    created_at = row.get("created_at")
    if isinstance(created_at, datetime):
        created_at = created_at.timestamp()
    return {**row, "id": str(row.get("id")), "created_at": created_at}


def push_live_alert(alert_data: dict):
    """Push a new alert from the ZMQ bridge into the in-memory buffer."""
    alert_entry = {
        "id": f"live-{len(_live_alerts) + 1}",
        "alert_type": alert_data.get("anomaly_reason", "UNKNOWN")[:30] if alert_data.get("anomaly_reason") else "ANOMALY",
        "severity": "critical" if "Hypoxia" in str(alert_data.get("anomaly_reason", "")) or "Heatwave" in str(alert_data.get("anomaly_reason", "")) else "high",
        "sensor_id": alert_data.get("sensor_id", "UNKNOWN"),
        "lat": alert_data.get("lat", 0.0),
        "lon": alert_data.get("lon", 0.0),
        "message": alert_data.get("anomaly_reason", "Anomaly detected"),
        "created_at": alert_data.get("timestamp", time.time()),
        "acknowledged": False,
    }
    _live_alerts.insert(0, alert_entry)
    if len(_live_alerts) > MAX_LIVE_ALERTS:
        _live_alerts.pop()


@router.get("/alerts/recent")
async def get_recent_alerts(
    limit: int = Query(20, ge=1, le=100, description="Number of alerts to return"),
    severity: Optional[str] = Query(None, description="Filter by severity: low, medium, high, critical"),
):
    try:
        db_alerts = [_normalize_db_alert(a) for a in await queries.get_recent_alerts(limit=limit, severity=severity)]
        all_alerts = _live_alerts + db_alerts
        source = "database"
    except Exception as e:
        logger.warning(f"DB unavailable for /alerts/recent, using sample fallback: {e}")
        all_alerts = _live_alerts + SAMPLE_ALERTS
        if severity:
            all_alerts = [a for a in all_alerts if a["severity"] == severity]
        source = "sample_fallback"

    all_alerts.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return {
        "status": "ok", "source": source,
        "count": len(all_alerts[:limit]), "alerts": all_alerts[:limit],
    }


@router.get("/alerts/active")
async def get_active_alerts():
    try:
        db_alerts = [_normalize_db_alert(a) for a in await queries.get_active_alerts()]
        active = _live_alerts + db_alerts
        source = "database"
    except Exception as e:
        logger.warning(f"DB unavailable for /alerts/active, using sample fallback: {e}")
        all_alerts = _live_alerts + SAMPLE_ALERTS
        active = [a for a in all_alerts if not a.get("acknowledged", False)]
        source = "sample_fallback"

    active.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    return {"status": "ok", "source": source, "count": len(active), "alerts": active}


@router.post("/alerts")
async def create_alert(alert: AlertCreate):
    now = datetime.now(timezone.utc)
    try:
        await queries.insert_alert({
            "alert_type": alert.alert_type, "severity": alert.severity,
            "sensor_id": alert.sensor_id, "lat": alert.lat, "lon": alert.lon,
            "message": alert.message, "created_at": now, "acknowledged": False,
        })
        persisted = True
    except Exception as e:
        logger.warning(f"Failed to persist alert to DB, keeping in-memory only: {e}")
        persisted = False

    alert_entry = {
        "id": f"live-{len(_live_alerts) + 1}", "alert_type": alert.alert_type,
        "severity": alert.severity, "sensor_id": alert.sensor_id,
        "lat": alert.lat, "lon": alert.lon, "message": alert.message,
        "created_at": now.timestamp(), "acknowledged": False,
    }
    _live_alerts.insert(0, alert_entry)
    logger.info(f"⚠️ New alert created (persisted={persisted}): [{alert.severity}] {alert.message}")
    return {"status": "created", "persisted": persisted, "alert": alert_entry}


@router.get("/alerts/stats")
async def get_alert_stats():
    try:
        stats = await queries.get_alert_stats()
        stats["active"] += len(_live_alerts)
        stats["total"] += len(_live_alerts)
        for a in _live_alerts:
            stats["by_severity"][a["severity"]] = stats["by_severity"].get(a["severity"], 0) + 1
            stats["by_type"][a["alert_type"]] = stats["by_type"].get(a["alert_type"], 0) + 1
        return {"status": "ok", "source": "database", "stats": stats}
    except Exception as e:
        logger.warning(f"DB unavailable for /alerts/stats, using sample fallback: {e}")
        all_alerts = _live_alerts + SAMPLE_ALERTS
        stats = {
            "total": len(all_alerts),
            "active": sum(1 for a in all_alerts if not a.get("acknowledged", False)),
            "by_severity": {}, "by_type": {},
        }
        for a in all_alerts:
            sev = a.get("severity", "unknown")
            atype = a.get("alert_type", "unknown")
            stats["by_severity"][sev] = stats["by_severity"].get(sev, 0) + 1
            stats["by_type"][atype] = stats["by_type"].get(atype, 0) + 1
        return {"status": "ok", "source": "sample_fallback", "stats": stats}