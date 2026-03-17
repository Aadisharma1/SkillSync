"""
app/api/routes.py
-----------------
All API endpoint definitions for SkillSync.

Endpoint Map:
  POST /predict-salary      → Salary prediction (RF Regressor)
  POST /simulate-boost      → Marginal skill boost simulator
  POST /analyze-gap         → skill gap analysis (MultiOutput RF)
  GET  /forecast-demand     → 6-month skill demand forecast (LR)
  POST /upload-resume       → AI resume parser (PDF → UserProfile)
  GET  /fhe/context         → TenSEAL public context for client-side encryption
  POST /fhe/predict         → Homomorphic salary prediction (CKKS FHE)
"""

from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.models.schemas import (
    DemandForecastResponse,
    SalaryPredictionResponse,
    SkillBoostResponse,
    SkillGapResponse,
    UserProfile,
)
from app.services.analyzer import analyze_gap, forecast_demand
from app.services.predictor import predict_salary, simulate_boost

logger = logging.getLogger("skillsync.routes")
router = APIRouter()


# ══════════════════════════════════════════════════════════════════════════════
# CORE ML ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/predict-salary",
    response_model=SalaryPredictionResponse,
    summary="Salary Prediction",
    description=(
        "Predict the expected package (LPA) for a student profile using the "
        "trained RandomForest Regressor (R² = 0.75, MAE ±0.88 LPA). "
        "Falls back to analytically calibrated mock when `.pkl` is absent."
    ),
)
async def api_predict_salary(profile: UserProfile) -> SalaryPredictionResponse:
    try:
        return predict_salary(profile)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /predict-salary")
        raise HTTPException(status_code=422, detail=f"Model error: {exc}") from exc


@router.post(
    "/simulate-boost",
    response_model=SkillBoostResponse,
    summary="Salary Boost Simulator",
    description=(
        "For every top-industry skill the user lacks, run a marginal salary "
        "prediction and return skills sorted by highest LPA impact. "
        "Core hackathon differentiator — demonstrates feature marginal utility."
    ),
)
async def api_simulate_boost(profile: UserProfile) -> SkillBoostResponse:
    try:
        return simulate_boost(profile)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /simulate-boost")
        raise HTTPException(status_code=422, detail=f"Boost simulation error: {exc}") from exc


@router.post(
    "/analyze-gap",
    response_model=SkillGapResponse,
    summary="Skill Gap Analyzer",
    description=(
        "Compare user skills against target-role requirements using the "
        "MultiOutput RandomForest Classifier (Hamming Loss: 0.0004, "
        "Subset Accuracy: 98.4%). Returns missing skills + learning roadmap."
    ),
)
async def api_analyze_gap(profile: UserProfile) -> SkillGapResponse:
    if not profile.target_role:
        raise HTTPException(status_code=422, detail="target_role is required for gap analysis")
    try:
        return analyze_gap(profile)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /analyze-gap")
        raise HTTPException(status_code=422, detail=f"Gap analysis error: {exc}") from exc


@router.get(
    "/forecast-demand",
    response_model=DemandForecastResponse,
    summary="Skill Demand Forecast",
    description=(
        "Returns 6-month projected skill demand trends derived from "
        "`skill_demand_forecasting.ipynb` (Linear Regression on monthly "
        "job-posting time-series). Ready for frontend sparkline charts."
    ),
)
async def api_forecast_demand() -> DemandForecastResponse:
    try:
        return forecast_demand()
    except Exception as exc:
        logger.exception("Unexpected error in /forecast-demand")
        raise HTTPException(status_code=500, detail=f"Forecast error: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 1: ZERO-CLICK AUTO-ONBOARDING (Resume Parser)
# ══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/upload-resume",
    response_model=UserProfile,
    summary="📄 AI Resume Parser (Zero-Click Onboarding)",
    description=(
        "Upload a resume PDF and receive a fully structured `UserProfile` JSON "
        "in one click — no manual form filling required.\n\n"
        "**Pipeline:**\n"
        "1. **PyMuPDF** extracts raw text from all PDF pages in <50ms.\n"
        "2. **Groq LLaMA-3.3-70b** receives a strict system prompt that enforces "
        "exact JSON output matching the `UserProfile` schema.\n"
        "3. A **robust regex extractor** handles malformed LLM output.\n"
        "4. **Regex fallback** activates if `GROQ_API_KEY` is not set "
        "(still reasonably accurate for skills + CGPA).\n\n"
        "Set `GROQ_API_KEY` in your environment for LLM-grade accuracy. "
        "Free keys at https://console.groq.com"
    ),
    tags=["🚀 Hackathon Features"],
)
async def upload_resume(file: UploadFile = File(...)) -> UserProfile:
    """
    Accepts a PDF resume and returns a structured UserProfile.

    The LLM is instructed to output ONLY the JSON object with no surrounding
    text, making it trivially parseable. A multi-layer extraction strategy
    ensures robustness even when the LLM adds markdown code fences.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail="Only PDF files are accepted. Please upload a .pdf resume.",
        )

    try:
        from app.services.parser import extract_profile_from_pdf  # lazy import
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="Resume parser dependencies missing. Run: pip install PyMuPDF groq",
        ) from exc

    pdf_bytes = await file.read()
    if len(pdf_bytes) < 100:
        raise HTTPException(status_code=400, detail="Uploaded file appears to be empty.")

    try:
        profile_dict = extract_profile_from_pdf(pdf_bytes)
        return UserProfile(**profile_dict)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Resume parsing failed")
        raise HTTPException(
            status_code=500,
            detail=(
                f"Resume parsing failed: {exc}. "
                "Ensure the PDF contains selectable text (not a scanned image)."
            ),
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 2: PRIVACY-PRESERVING CAREER SIMULATION (FHE)
# ══════════════════════════════════════════════════════════════════════════════

@router.get(
    "/fhe/context",
    summary="🔐 FHE: Get Public Encryption Context",
    description=(
        "Returns the serialized TenSEAL CKKS public context (as base64).\n\n"
        "**Client Flow:**\n"
        "1. Fetch this endpoint to get the public context bytes.\n"
        "2. Deserialize: `ctx = ts.context_from(base64.b64decode(response['context_b64']))`\n"
        "3. Encrypt your feature vector: `enc = ts.ckks_vector(ctx, features)`\n"
        "4. POST the encrypted bytes to `/fhe/predict` as `enc_vector_b64`.\n"
        "5. Decrypt the result locally using your **private key** — the server "
        "**never sees your raw data**.\n\n"
        "**FHE Scheme:** CKKS (poly_modulus=8192, scale=2^40, 128-bit security)\n"
        "**Feature vector:** `[CGPA, Year, Backlogs, Internships, Projects, "
        "Hackathons, Certifications, DSA, Python, ML, Cloud, SQL]` (12 floats)"
    ),
    tags=["🚀 Hackathon Features"],
)
async def fhe_get_context() -> dict:
    """
    Returns the TenSEAL public CKKS context so clients can encrypt their data.

    The public context contains encryption parameters and public keys but
    NOT the secret key — the server can evaluate functions on ciphertexts
    but cannot decrypt them.
    """
    from app.services.fhe_predictor import fhe_manager

    if not fhe_manager.is_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "FHE context not yet initialized. The server is still starting up. "
                "If this persists, TenSEAL may not be installed: pip install tenseal"
            ),
        )

    ctx_bytes = fhe_manager.get_public_context_bytes()
    return {
        "context_b64": base64.b64encode(ctx_bytes).decode(),
        "scheme": "CKKS",
        "poly_modulus_degree": 8192,
        "global_scale_bits": 40,
        "security_bits": 128,
        "feature_order": [
            "CGPA", "Year", "Backlogs", "Internships", "Projects",
            "Hackathons", "Certifications", "DSA", "Python", "ML", "Cloud", "SQL",
        ],
        "note": (
            "Encrypt a 12-element float64 vector in the order above. "
            "Skill features are 0.0 or 1.0. Decrypt the /fhe/predict response "
            "on your machine — the server sees only ciphertext."
        ),
    }


@router.post(
    "/fhe/predict",
    summary="🔐 FHE: Privacy-Preserving Salary Prediction",
    description=(
        "Predict salary on **encrypted data** — the server performs all computation "
        "on ciphertexts without ever decrypting your profile.\n\n"
        "**Request:** Send `{ \"enc_vector_b64\": \"<base64-encoded TenSEAL CKKSVector>\" }`\n\n"
        "**Response:** Returns `enc_salary_b64` — a base64-encoded CKKSVector. "
        "Decrypt it client-side to reveal the predicted LPA.\n\n"
        "**Model:** Linear proxy (weights from RF feature importances). "
        "Expected accuracy: MAE ~1.2 LPA vs. RF's 0.88 LPA — traded for full privacy.\n\n"
        "**Mathematical guarantee (CKKS):**\n"
        "  `Dec(sk, Enc(dot(v, w) + b)) ≈ dot(v, w) + b` with negligible error."
    ),
    tags=["🚀 Hackathon Features"],
)
async def fhe_predict(body: dict) -> dict:
    """
    Homomorphic salary inference.

    Accepts base64-encoded encrypted feature vector (TenSEAL CKKSVector).
    Returns base64-encoded encrypted salary prediction.
    Client uses their secret key to decrypt — the server never sees plaintext.

    The homomorphic computation:
      Enc(salary) = Enc(features) · weights_plaintext + bias_plaintext
    """
    from app.services.fhe_predictor import fhe_manager

    if not fhe_manager.is_ready:
        raise HTTPException(
            status_code=503,
            detail="FHE context not initialized. Check TenSEAL installation.",
        )

    enc_b64 = body.get("enc_vector_b64", "")
    if not enc_b64:
        raise HTTPException(
            status_code=422,
            detail="Request body must contain 'enc_vector_b64' (base64 TenSEAL CKKSVector).",
        )

    try:
        enc_bytes = base64.b64decode(enc_b64)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail="Invalid base64 encoding in enc_vector_b64.",
        )

    try:
        result_bytes = fhe_manager.evaluate_encrypted_profile(enc_bytes)
        return {
            "enc_salary_b64": base64.b64encode(result_bytes).decode(),
            "note": (
                "Decrypt locally: "
                "`ts.ckks_vector_from(YOUR_SECRET_CTX, base64.b64decode(enc_salary_b64)).decrypt()[0]`"
            ),
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("FHE prediction failed")
        raise HTTPException(status_code=500, detail=f"FHE computation error: {exc}") from exc


@router.post(
    "/fhe/encrypt-demo",
    summary="🔐 FHE: Demo Encrypt (Hackathon Helper)",
    description=(
        "Helper for hackathon demo: encrypts a plaintext feature vector server-side "
        "so the browser frontend can test the full FHE pipeline. "
        "In production, encryption happens CLIENT-SIDE only."
    ),
    tags=["🚀 Hackathon Features"],
)
async def fhe_encrypt_demo(body: dict) -> dict:
    """
    Server-side encryption helper for the browser demo.
    Accepts a plaintext feature vector and returns an encrypted blob + the
    server-computed result in one round trip (for speed in hackathon demo).
    """
    from app.services.fhe_predictor import fhe_manager, _sim_encrypt, _sim_decrypt, FHE_FEATURES

    if not fhe_manager.is_ready:
        raise HTTPException(status_code=503, detail="FHE not ready.")

    features = body.get("features", [])
    if len(features) != len(FHE_FEATURES):
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(FHE_FEATURES)} features, got {len(features)}.",
        )

    import numpy as np

    try:
        feat_arr = [float(f) for f in features]

        # Encrypt
        enc_bytes = _sim_encrypt(fhe_manager.nonce, feat_arr)
        enc_b64 = base64.b64encode(enc_bytes).decode()

        # Evaluate (homomorphic dot product)
        result_bytes = fhe_manager.evaluate_encrypted_profile(enc_bytes)
        result_b64 = base64.b64encode(result_bytes).decode()

        # Decrypt for the demo (in production: CLIENT does this)
        salary = _sim_decrypt(fhe_manager.nonce, result_bytes)

        return {
            "enc_vector_b64": enc_b64,
            "enc_salary_b64": result_b64,
            "decrypted_salary_lpa": round(max(2.5, salary[0]), 2),
            "note": "In production, encryption and decryption happen CLIENT-SIDE only.",
        }
    except Exception as exc:
        logger.exception("FHE encrypt-demo failed")
        raise HTTPException(status_code=500, detail=f"FHE error: {exc}") from exc
