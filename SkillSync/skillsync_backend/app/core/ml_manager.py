"""
app/core/ml_manager.py
-----------------------
Singleton ML model manager.
Loads all .pkl files once at startup; sets models_loaded=False on any
failure so every endpoint can fall back to rich mock data — frontend
development is never blocked even if models are absent.
"""

import json
import logging
import warnings

import joblib
import numpy as np
import pandas as pd

from app.core.config import (
    ALL_SKILLS_SORTED,
    BOOST_CANDIDATE_SKILLS,
    FEATURE_COLUMNS_PATH,
    SALARY_BRANCH_COLS,
    SALARY_MODEL_PATH,
    SALARY_NUMERIC_COLS,
    SALARY_SKILL_COLS,
    SKILL_GAP_META_PATH,
    SKILL_GAP_MODEL_PATH,
    VALID_ROLES,
)

logger = logging.getLogger("skillsync.ml_manager")


class MLModelManager:
    """Singleton that owns all loaded ML artifacts."""

    _instance: "MLModelManager | None" = None

    def __new__(cls) -> "MLModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    # ── Startup ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load every artifact. Called once from the FastAPI lifespan."""
        if self._initialised:
            return

        self.salary_model = None
        self.skill_gap_model = None
        self.feature_columns: list[str] = []
        self.skill_gap_meta: dict = {}
        self.models_loaded = False

        errors: list[str] = []

        # ── salary model ──
        try:
            self.salary_model = joblib.load(SALARY_MODEL_PATH)
            logger.info("✅  Salary model loaded from %s", SALARY_MODEL_PATH)
        except Exception as exc:
            errors.append(f"salary_model: {exc}")

        # ── skill-gap model ──
        try:
            self.skill_gap_model = joblib.load(SKILL_GAP_MODEL_PATH)
            logger.info("✅  Skill-gap model loaded from %s", SKILL_GAP_MODEL_PATH)
        except Exception as exc:
            errors.append(f"skill_gap_model: {exc}")

        # ── feature columns (used by salary model alignment) ──
        try:
            self.feature_columns = list(joblib.load(FEATURE_COLUMNS_PATH))
            logger.info("✅  Feature columns loaded (%d cols)", len(self.feature_columns))
        except Exception as exc:
            errors.append(f"feature_columns: {exc}")
            # Fall back to the canonical order we know from the notebook
            self.feature_columns = (
                SALARY_NUMERIC_COLS + SALARY_SKILL_COLS + SALARY_BRANCH_COLS
            )

        # ── skill-gap metadata ──
        try:
            with open(SKILL_GAP_META_PATH, encoding="utf-8") as fh:
                self.skill_gap_meta = json.load(fh)
            logger.info("✅  Skill-gap meta loaded")
        except Exception as exc:
            errors.append(f"skill_gap_meta: {exc}")

        if errors:
            logger.warning(
                "\n"
                + "=" * 70
                + "\n"
                + "⚠️  ONE OR MORE ML ARTIFACTS FAILED TO LOAD — USING MOCK DATA\n"
                + "\n".join(f"    • {e}" for e in errors)
                + "\n"
                + "=" * 70
            )
            self.models_loaded = False
        else:
            self.models_loaded = True
            logger.info("🚀  All ML models loaded successfully — live predictions active")

        self._initialised = True

    # ── Feature alignment: Salary model ──────────────────────────────────────

    def build_salary_feature_vector(
        self,
        cgpa: float,
        year: int,
        backlogs: int,
        internships: int,
        projects: int,
        hackathons: int,
        certifications: int,
        current_skills: list[str],
        branch: str,
    ) -> pd.DataFrame:
        """
        Build a single-row DataFrame that exactly matches the column order
        stored in feature_columns.pkl.

        Salary model columns (from notebook):
            CGPA, Year, Backlogs, DSA, Python, Java, Web_Dev, ML,
            Cloud, DevOps, SQL, Internships, Projects, Hackathons,
            Certifications, Branch_ECE, Branch_IT, Branch_MECH
        """
        # Start with all zeros
        row: dict[str, float] = {col: 0.0 for col in self.feature_columns}

        # Numeric
        row["CGPA"] = float(cgpa)
        row["Year"] = float(year)
        row["Backlogs"] = float(backlogs)
        row["Internships"] = float(internships)
        row["Projects"] = float(projects)
        row["Hackathons"] = float(hackathons)
        row["Certifications"] = float(certifications)

        # Skill bits  (frontend sends salary-model skill keys directly)
        for sk in SALARY_SKILL_COLS:
            if sk in current_skills:
                row[sk] = 1.0

        # Branch one-hot
        branch_col = f"Branch_{branch.upper()}"
        if branch_col in row:
            row[branch_col] = 1.0

        df = pd.DataFrame([row])[self.feature_columns]  # enforce column order
        return df

    # ── Feature alignment: Skill-gap model ───────────────────────────────────

    def build_gap_feature_vector(
        self, current_skills: list[str], target_role: str
    ) -> pd.DataFrame:
        """
        Build the feature vector for the MultiOutput gap model:
          - One-hot encode Target_Role  (10 columns)
          - Binary encode 55 skill bits (same order as ALL_SKILLS_SORTED)
        Total: 10 + 55 = 65 columns — matches X_train shape in notebook.
        """
        # Role one-hot
        role_dict: dict[str, int] = {}
        for role in VALID_ROLES:
            col = f"Target_Role_{role}"
            role_dict[col] = 1 if role == target_role else 0

        # Skill bits
        skill_dict: dict[str, int] = {sk: 0 for sk in ALL_SKILLS_SORTED}
        for sk in current_skills:
            if sk in skill_dict:
                skill_dict[sk] = 1

        row = {**role_dict, **skill_dict}
        return pd.DataFrame([row])


# Module-level singleton — import this everywhere
ml_manager = MLModelManager()
