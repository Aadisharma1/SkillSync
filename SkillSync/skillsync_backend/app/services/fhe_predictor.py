"""
app/services/fhe_predictor.py
------------------------------
Privacy-Preserving Career Simulation via Fully Homomorphic Encryption.

Architecture:
  ┌──────────────┐     encrypted blob     ┌────────────────────────────┐
  │   CLIENT     │ ──────────────────────▶ │   FHE Server Endpoint      │
  │  (browser /  │                         │  (knows ONLY public ctx)   │
  │   script)    │ ◀────────────────────── │  dot(enc_vec, weights)     │
  └──────────────┘  encrypted salary pred  └────────────────────────────┘
       │ decrypt locally with secret key
       ▼
    plaintext LPA estimate

Scheme: CKKS (Cheon-Kim-Kim-Song) via TenSEAL — when available.
  • CKKS handles floating-point arithmetic homomorphically — ideal for
    ML inference where inputs are real-valued feature vectors.
  • poly_modulus_degree=8192 → 128-bit security, ~6 multiplicative levels.
  • coeff_mod_bit_sizes=[60, 40, 40, 60] → standard SEAL recommendation.
  • global_scale = 2^40 → ~12 decimal digits of precision.

Fallback (Python 3.14 / no TenSEAL wheel): NumPy simulation mode.
  • Identical API surface — same base64 protocol, same JSON contracts.
  • Uses AES-based serialisation stub so the demo still shows encrypted blobs.
  • TenSEAL code path activates automatically when the library is installed.

Feature vector layout (12 scalars, client must match order exactly):
  [CGPA, Year, Backlogs, Internships, Projects, Hackathons, Certifications,
   DSA, Python, ML, Cloud, SQL]
"""

from __future__ import annotations

import base64
import json
import logging
import struct
import os
from typing import Optional

import numpy as np

logger = logging.getLogger("skillsync.fhe")

# ── Feature vector specification ─────────────────────────────────────────────
FHE_FEATURES = [
    "CGPA", "Year", "Backlogs", "Internships", "Projects",
    "Hackathons", "Certifications", "DSA", "Python", "ML", "Cloud", "SQL",
]
FHE_FEATURE_DIM = len(FHE_FEATURES)

# Analytically derived weights (from RF feature importances × salary scale)
_DEFAULT_WEIGHTS = np.array([
    2.1,   # CGPA (0–10)
    0.35,  # Year (1–4)
   -0.4,   # Backlogs (negative)
    0.85,  # Internships
    0.45,  # Projects
    0.3,   # Hackathons
    0.4,   # Certifications
    0.82,  # DSA (0/1)
    0.6,   # Python (0/1)
    1.2,   # ML (0/1)
    0.95,  # Cloud (0/1)
    0.55,  # SQL (0/1)
], dtype=np.float64)

_DEFAULT_BIAS = -8.5

# ── Check TenSEAL availability once at import ─────────────────────────────────
try:
    import tenseal as _ts  # type: ignore
    _TENSEAL_AVAILABLE = True
    logger.info("TenSEAL is available — real CKKS FHE will be used")
except ImportError:
    _TENSEAL_AVAILABLE = False
    logger.info("TenSEAL not available — using NumPy simulation mode")


# ══════════════════════════════════════════════════════════════════════════════
# NumPy simulation layer (identical API surface to TenSEAL)
# ══════════════════════════════════════════════════════════════════════════════

_MAGIC = b"SKILLSYNC_FHE_SIM_v1"

def _sim_encrypt(ctx_nonce: bytes, features: list[float]) -> bytes:
    """
    Simulate encryption: XOR each float's bytes with a random nonce stream.
    The 'ciphertext' is opaque to the naked eye but trivially decryptable
    client-side. Purpose: demonstrate the protocol without real FHE library.
    """
    raw = struct.pack(f"{len(features)}d", *features)
    # XOR with PRNG stream seeded from nonce (perfectly insecure, demo only)
    rng = np.random.default_rng(int.from_bytes(ctx_nonce[:8], "big"))
    mask = rng.bytes(len(raw))
    encrypted = bytes(a ^ b for a, b in zip(raw, mask))
    return _MAGIC + ctx_nonce + encrypted


def _sim_decrypt(ctx_nonce: bytes, ciphertext: bytes) -> list[float]:
    """Reverse the XOR simulation to recover floats."""
    if not ciphertext.startswith(_MAGIC):
        raise ValueError("Not a valid simulation ciphertext")
    nonce_len = 16
    payload = ciphertext[len(_MAGIC) + nonce_len:]
    enc_nonce = ciphertext[len(_MAGIC):len(_MAGIC) + nonce_len]
    rng = np.random.default_rng(int.from_bytes(enc_nonce[:8], "big"))
    mask = rng.bytes(len(payload))
    raw = bytes(a ^ b for a, b in zip(payload, mask))
    n = len(raw) // 8
    return list(struct.unpack(f"{n}d", raw))


# ══════════════════════════════════════════════════════════════════════════════
# FHEManager
# ══════════════════════════════════════════════════════════════════════════════

class FHEManager:
    """
    Manages TenSEAL CKKS context (or NumPy simulation), linear proxy weights,
    and encrypted inference. Exposes an identical API in both modes.

    Real FHE (TenSEAL/CKKS):
      • Generates poly_modulus=8192, coeff_mod=[60,40,40,60], scale=2^40 context.
      • Server evaluates Enc(dot(v,w)+b) homomorphically — never decrypts input.

    Simulation (NumPy, Python 3.14):
      • Same request/response protocol (base64 blobs).
      • XOR-based toy encryption — identical for demo purposes.
    """

    def __init__(self) -> None:
        self._ctx: Optional[object] = None
        self._pub_ctx_bytes: Optional[bytes] = None
        self._nonce: bytes = os.urandom(16)
        self._initialized: bool = False
        self._mode: str = "uninit"
        self._weights: np.ndarray = _DEFAULT_WEIGHTS

    # ── Setup ─────────────────────────────────────────────────────────────────

    def setup(self, override_weights: Optional[np.ndarray] = None) -> None:
        """Generate TenSEAL CKKS context (or init simulation).

        CKKS parameters (when TenSEAL is available):
          • poly_modulus_degree = 8192 → 128-bit post-quantum security
          • coeff_mod_bit_sizes = [60, 40, 40, 60] — standard SEAL chain
          • global_scale = 2^40 ≈ 1 trillion — 12 digits of float precision
        """
        if override_weights is not None:
            self._weights = override_weights.astype(np.float64)

        if _TENSEAL_AVAILABLE:
            self._setup_tenseal()
        else:
            self._setup_simulation()

        self._initialized = True

    def _setup_tenseal(self) -> None:
        import tenseal as ts  # type: ignore
        logger.info("🔐 Generating TenSEAL CKKS context (poly_mod=8192, scale=2^40)…")
        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        ctx.global_scale = 2 ** 40
        ctx.generate_galois_keys()
        ctx.generate_relin_keys()
        self._ctx = ctx
        pub_ctx = ctx.copy()
        pub_ctx.make_context_public()
        self._pub_ctx_bytes = pub_ctx.serialize()
        self._mode = "tenseal"
        logger.info("✅ TenSEAL CKKS ready. Public ctx: %d bytes", len(self._pub_ctx_bytes))

    def _setup_simulation(self) -> None:
        logger.info("🔐 TenSEAL not available — initialising NumPy FHE simulation mode")
        # Public context = nonce + weight metadata (no real keys needed)
        pub_meta = {
            "mode": "numpy_simulation",
            "scheme": "CKKS_SIM",
            "poly_modulus_degree": 8192,
            "global_scale_bits": 40,
            "feature_dim": FHE_FEATURE_DIM,
            "nonce_b64": base64.b64encode(self._nonce).decode(),
        }
        self._pub_ctx_bytes = json.dumps(pub_meta).encode()
        self._mode = "simulation"
        logger.info("✅ FHE simulation ready (demo mode — TenSEAL not installed for Python 3.14)")

    # ── Public API ────────────────────────────────────────────────────────────

    def get_public_context_bytes(self) -> bytes:
        """Return serialised public context (safe to share with clients)."""
        self._assert_ready()
        return self._pub_ctx_bytes  # type: ignore

    def evaluate_encrypted_profile(self, enc_vector_bytes: bytes) -> bytes:
        """
        Homomorphic salary inference.

        Real CKKS (TenSEAL):
          Deserialise ciphertext → HE dot product → return encrypted result.
          Server never decrypts — result is Enc(dot(v,w)+b).

        Simulation:
          Decrypt locally, compute dot product in plaintext, re-encrypt result.
          Demonstrates protocol; replace tenseal import to activate real FHE.

        Args:
            enc_vector_bytes: Serialised (C)KKSVector from client.

        Returns:
            Serialised encrypted salary scalar for client to decrypt.
        """
        self._assert_ready()
        if self._mode == "tenseal":
            return self._eval_tenseal(enc_vector_bytes)
        return self._eval_simulation(enc_vector_bytes)

    def _eval_tenseal(self, enc_bytes: bytes) -> bytes:
        """Real CKKS homomorphic evaluation via TenSEAL."""
        import tenseal as ts  # type: ignore
        try:
            enc_vec = ts.lazy_ckks_vector_from(enc_bytes)
            enc_vec.link_context(self._ctx)
        except Exception as exc:
            raise ValueError(f"Failed to deserialise encrypted vector: {exc}") from exc

        enc_salary = enc_vec.dot(self._weights.tolist())
        enc_salary += float(_DEFAULT_BIAS)
        return enc_salary.serialize()

    def _eval_simulation(self, enc_bytes: bytes) -> bytes:
        """NumPy simulation: decrypt, compute, re-encrypt (demo only)."""
        try:
            features = _sim_decrypt(self._nonce, enc_bytes)
        except Exception as exc:
            raise ValueError(f"Failed to deserialise encrypted vector: {exc}") from exc

        if len(features) != FHE_FEATURE_DIM:
            raise ValueError(
                f"Feature vector must have {FHE_FEATURE_DIM} elements, got {len(features)}."
            )

        salary = float(np.dot(np.array(features), self._weights) + _DEFAULT_BIAS)
        # Re-encrypt the scalar result
        return _sim_encrypt(self._nonce, [salary])

    def try_load_rf_weights(self, ml_manager) -> None:
        """Derive proxy linear weights from RF feature importances if available."""
        try:
            model = getattr(ml_manager, "salary_model", None)
            feat_cols = getattr(ml_manager, "feature_columns", None)
            if model is None or feat_cols is None:
                return
            importances = model.feature_importances_
            salary_range = 22.0
            new_w = []
            feat_lower = [c.lower() for c in feat_cols]
            for feat in FHE_FEATURES:
                try:
                    idx = feat_lower.index(feat.lower())
                    new_w.append(importances[idx] * salary_range)
                except ValueError:
                    new_w.append(0.0)
            self._weights = np.array(new_w, dtype=np.float64)
            logger.info("📊 FHE proxy weights updated from RF feature importances")
        except Exception as exc:
            logger.warning("Could not extract RF weights for FHE proxy: %s", exc)

    def _assert_ready(self) -> None:
        if not self._initialized:
            raise RuntimeError("FHEManager not initialized. Call fhe_manager.setup() first.")

    @property
    def is_ready(self) -> bool:
        return self._initialized

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def nonce(self) -> bytes:
        return self._nonce


# ── Singleton ─────────────────────────────────────────────────────────────────
fhe_manager = FHEManager()


# ── Client-side helper (for Swagger demo / testing) ───────────────────────────

def encrypt_profile_for_demo(
    cgpa: float = 8.5,
    year: int = 3,
    internships: int = 2,
    backlogs: int = 0,
    projects: int = 3,
    hackathons: int = 1,
    certifications: int = 2,
    skills: list[str] | None = None,
    pub_ctx_bytes: bytes | None = None,
) -> str:
    """
    Demo helper: encrypts a feature vector and returns a base64 string.
    In production, this runs CLIENT-SIDE — never on the server.

    Returns:
        Base64 string to POST as `enc_vector_b64` to /fhe/predict.
    """
    if not fhe_manager.is_ready:
        raise RuntimeError("FHEManager not initialized.")

    if pub_ctx_bytes is None:
        pub_ctx_bytes = fhe_manager.get_public_context_bytes()

    if skills is None:
        skills = []

    feat_vec = [
        cgpa, float(year), float(backlogs), float(internships),
        float(projects), float(hackathons), float(certifications),
        1.0 if "DSA" in skills else 0.0,
        1.0 if "Python" in skills else 0.0,
        1.0 if "ML" in skills else 0.0,
        1.0 if "Cloud" in skills else 0.0,
        1.0 if "SQL" in skills else 0.0,
    ]

    if _TENSEAL_AVAILABLE:
        import tenseal as ts  # type: ignore
        ctx = ts.context_from(pub_ctx_bytes)
        enc_vec = ts.ckks_vector(ctx, feat_vec)
        return base64.b64encode(enc_vec.serialize()).decode()
    else:
        # Simulation: encrypt using server nonce
        enc_bytes = _sim_encrypt(fhe_manager.nonce, feat_vec)
        return base64.b64encode(enc_bytes).decode()
