"""
app/main.py
-----------
FastAPI application entry-point.

Features:
  • Lifespan context manager loads ML models once at startup
  • Root URL (/) serves the SkillSync frontend HTML directly
  • Global exception handler converts dimension-mismatch errors → HTTP 422
  • CORS: allow_origins=["*"]
  • API routes prefixed at /api/v1
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

# Load .env file (for GROQ_API_KEY etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from app.api.routes import router
from app.core.ml_manager import ml_manager

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("skillsync")

# ── Paths ─────────────────────────────────────────────────────────────────────
# skillsync_backend/app/main.py  →  SkillSync/ (project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_HTML = _PROJECT_ROOT / "SkillSyncsample.html"


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models + FHE context on startup; release on shutdown."""
    import asyncio
    import threading

    logger.info("⚙️  SkillSync backend starting — loading ML models…")
    ml_manager.load()
    if ml_manager.models_loaded:
        logger.info("✅  All models live — real predictions active")
    else:
        logger.warning("⚠️  Models not loaded — endpoints return mock data")

    # Initialize FHE in background thread (TenSEAL key gen takes ~1-2s)
    def _init_fhe():
        try:
            from app.services.fhe_predictor import fhe_manager
            fhe_manager.setup()
            # Try to derive proxy weights from loaded RF model
            fhe_manager.try_load_rf_weights(ml_manager)
        except Exception as exc:
            logger.warning("⚠️  FHE init failed (tenseal probably not installed): %s", exc)

    fhe_thread = threading.Thread(target=_init_fhe, daemon=True)
    fhe_thread.start()

    logger.info("🌐  Frontend: http://localhost:8000")
    logger.info("📄  Swagger:  http://localhost:8000/docs")
    yield
    logger.info("🛑  SkillSync backend shutting down")



# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SkillSync API",
    version="2.0.0",
    description=(
        "AI-Powered Career Intelligence Platform\n\n"
        "4 ML models integrated:\n"
        "- RandomForest Classifier → Job Role Prediction (99.4% acc)\n"
        "- MultiOutput RandomForest → Skill Gap Analysis (98.4% acc)\n"
        "- RandomForest Regressor → Salary Prediction (R²=0.75)\n"
        "- Linear Regression → Skill Demand Forecasting\n\n"
        "All endpoints fall back gracefully when .pkl files are absent."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global exception handlers ─────────────────────────────────────────────────

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "type": "validation_error"},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    msg = str(exc)
    if "feature" in msg.lower() and "shape" in msg.lower():
        detail = (
            "Feature dimension mismatch — check skill key names match "
            "feature_columns.pkl exactly (e.g. 'Web_Dev' not 'WebDev')."
        )
    else:
        detail = f"Internal server error: {msg}"
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(status_code=422, content={"detail": detail})

# ── Root — serve the frontend HTML ────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the SkillSync HTML dashboard directly from the project root."""
    if _FRONTEND_HTML.exists():
        return FileResponse(str(_FRONTEND_HTML), media_type="text/html")
    # Fallback: redirect to Swagger if HTML not found
    return RedirectResponse(url="/docs")


@app.get("/ui", include_in_schema=False)
async def serve_frontend_alias():
    return RedirectResponse(url="/")

# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health() -> dict:
    return {
        "status": "ok",
        "models_loaded": ml_manager.models_loaded,
        "version": "2.0.0",
        "frontend": str(_FRONTEND_HTML),
    }

# ── API routes ────────────────────────────────────────────────────────────────

app.include_router(router, prefix="/api/v1", tags=["SkillSync ML"])

# ── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
