"""
BlueByte AI — FastAPI Application Entry Point
Initializes the API server, registers routes, and manages lifespan events.
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from server.api.routes import ocean, predictions, alerts, chat
from server.api.websocket_manager import router as ws_router
from server.api.zmq_bridge import ZMQBridge

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("BlueByte-API")

# Global ZMQ bridge instance
zmq_bridge: ZMQBridge | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    global zmq_bridge
    logger.info("🌊 BlueByte AI API Server starting up...")

    # Initialize database
    try:
        from db.connection import init_db
        await init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.warning(f"⚠️ Database init skipped (run `python -m db.seed_data` first): {e}")

    # Start ZMQ bridge in background
    zmq_bridge = ZMQBridge()
    bridge_task = asyncio.create_task(zmq_bridge.start())
    logger.info("✅ ZMQ Bridge started")

    yield

    # Shutdown
    logger.info("🛑 Shutting down BlueByte AI API Server...")
    if zmq_bridge:
        zmq_bridge.stop()
    bridge_task.cancel()


app = FastAPI(
    title="BlueByte AI",
    description="AI-Driven Unified Data Platform for Oceanographic, Fisheries & Molecular Biodiversity Insight",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for hackathon demo.
# FIXED: allow_origins=["*"] + allow_credentials=True is invalid per
# the CORS spec — browsers block credentialed requests against a
# wildcard origin, so this was silently blocking anything that sent
# cookies/Authorization with credentials:'include', even though the
# server thought it was allowing everything. The frontend doesn't
# actually send credentials anywhere (checked frontend/react_app/src/),
# so allow_credentials=False is the correct fix here — flip this back
# to True (and swap allow_origins for an explicit origin list) only if
# cookie-based auth gets added later, since wildcard + credentials=True
# can never work together.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routes
app.include_router(ocean.router, prefix="/api/v1", tags=["Ocean Data"])
app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
app.include_router(alerts.router, prefix="/api/v1", tags=["Alerts"])
app.include_router(chat.router, prefix="/api/v1", tags=["AI Chatbot (GraphRAG)"])
app.include_router(ws_router, tags=["WebSocket"])

# Serve frontend static files
client_dir = os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "react_app")
if os.path.isdir(client_dir):
    app.mount("/", StaticFiles(directory=client_dir, html=True), name="frontend")


@app.get("/health", tags=["System"])
async def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "BlueByte AI",
        "version": "1.0.0",
        "zmq_bridge_active": zmq_bridge is not None and zmq_bridge.running,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.api.main:app", host="0.0.0.0", port=8000, reload=True)