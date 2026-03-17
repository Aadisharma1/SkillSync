"""
app/models/schemas.py
---------------------
Pydantic request / response models with Swagger-UI example values.
"""

from typing import Any
from pydantic import BaseModel, Field


# ════════════════════════════════════════════════
#  INPUT
# ════════════════════════════════════════════════

class UserProfile(BaseModel):
    """Common profile sent by the frontend for all prediction endpoints."""

    cgpa: float = Field(
        ..., ge=0.0, le=10.0,
        description="CGPA on a 10-point scale",
        json_schema_extra={"example": 8.5},
    )
    year: int = Field(
        2, ge=1, le=4,
        description="Current academic year (1–4)",
        json_schema_extra={"example": 3},
    )
    branch: str = Field(
        ...,
        description="Engineering branch: CSE | IT | ECE | MECH",
        json_schema_extra={"example": "CSE"},
    )
    backlogs: int = Field(
        0, ge=0,
        description="Number of active backlogs",
        json_schema_extra={"example": 0},
    )
    internships: int = Field(
        0, ge=0,
        description="Number of internships completed",
        json_schema_extra={"example": 2},
    )
    projects: int = Field(
        0, ge=0,
        description="Number of personal/academic projects",
        json_schema_extra={"example": 3},
    )
    hackathons: int = Field(
        0, ge=0,
        description="Number of hackathons participated in",
        json_schema_extra={"example": 1},
    )
    certifications: int = Field(
        0, ge=0,
        description="Number of professional certifications",
        json_schema_extra={"example": 2},
    )
    # Salary model skill keys  (binary — sent as ["DSA","Python","ML", ...])
    current_skills: list[str] = Field(
        default_factory=list,
        description=(
            "Active skills. For salary/boost: DSA|Python|Java|Web_Dev|ML|Cloud|DevOps|SQL. "
            "For gap analysis: any of the 55 gap-model skills."
        ),
        json_schema_extra={"example": ["Python", "ML", "SQL"]},
    )
    # Optional — used only by analyze-gap
    target_role: str | None = Field(
        None,
        description="Target role for skill-gap analysis",
        json_schema_extra={"example": "Data Scientist"},
    )


# ════════════════════════════════════════════════
#  OUTPUT — Salary
# ════════════════════════════════════════════════

class FeatureImportanceHint(BaseModel):
    feature: str
    importance: float
    your_value: Any


class SalaryPredictionResponse(BaseModel):
    predicted_lpa: float = Field(description="Predicted package in LPA")
    range_low: float = Field(description="Range lower bound (predicted_lpa - 0.9)")
    range_high: float = Field(description="Range upper bound (predicted_lpa + 0.9)")
    feature_importance_hints: list[FeatureImportanceHint] = Field(
        description="Top features and their contributions"
    )
    mock: bool = Field(False, description="True when models aren't loaded — demo data only")


# ════════════════════════════════════════════════
#  OUTPUT — Boost
# ════════════════════════════════════════════════

class BoostEntry(BaseModel):
    skill: str
    boost_lpa: float = Field(description="Marginal LPA increase from adding this skill")


class SkillBoostResponse(BaseModel):
    current_lpa: float
    boost_simulations: list[BoostEntry] = Field(
        description="Skills sorted by highest salary impact"
    )
    mock: bool = Field(False)


# ════════════════════════════════════════════════
#  OUTPUT — Gap analysis
# ════════════════════════════════════════════════

class LearningStep(BaseModel):
    skill: str
    tip: str


class SkillGapResponse(BaseModel):
    target_role: str
    suitability_score: float = Field(description="0–100 readiness percentage")
    skills_you_have: list[str]
    missing_skills: list[str]
    bonus_skills: list[str] = Field(description="Skills user has that aren't required")
    learning_roadmap: list[LearningStep]
    mock: bool = Field(False)


# ════════════════════════════════════════════════
#  OUTPUT — Demand forecast
# ════════════════════════════════════════════════

class DemandTrend(BaseModel):
    skill: str
    history: list[float] = Field(description="Monthly demand figures (Jan 2022 → now)")
    growth_pct: float = Field(description="6-month projected growth %")
    color: str = Field(description="Hex color for frontend charts")


class DemandForecastResponse(BaseModel):
    trends: list[DemandTrend]
    mock: bool = Field(False)
