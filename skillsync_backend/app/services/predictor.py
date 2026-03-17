"""
app/services/predictor.py
--------------------------
Salary prediction and boost simulation business logic.

Salary model features (notebook-derived order):
  CGPA, Year, Backlogs, DSA, Python, Java, Web_Dev, ML,
  Cloud, DevOps, SQL, Internships, Projects, Hackathons,
  Certifications, Branch_ECE, Branch_IT, Branch_MECH
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.core.config import BOOST_CANDIDATE_SKILLS, SALARY_SKILL_COLS
from app.core.ml_manager import ml_manager
from app.models.schemas import (
    BoostEntry,
    FeatureImportanceHint,
    SalaryPredictionResponse,
    SkillBoostResponse,
)

if TYPE_CHECKING:  # avoid circular at runtime
    from app.models.schemas import UserProfile

logger = logging.getLogger("skillsync.predictor")

# Feature importances from the notebook (approximate, for display)
_FEATURE_IMPORTANCE = {
    "CGPA": 0.233,
    "Internships": 0.178,
    "ML": 0.119,
    "DSA": 0.082,
    "Projects": 0.070,
    "Cloud": 0.068,
    "Certifications": 0.063,
    "Hackathons": 0.048,
    "Python": 0.040,
    "Year": 0.017,
}


# ── Internal helpers ──────────────────────────────────────────────────────────


def _run_salary_model(profile: "UserProfile") -> float:
    """Return raw salary prediction (LPA) for a profile dict."""
    df = ml_manager.build_salary_feature_vector(
        cgpa=profile.cgpa,
        year=profile.year,
        backlogs=profile.backlogs,
        internships=profile.internships,
        projects=profile.projects,
        hackathons=profile.hackathons,
        certifications=profile.certifications,
        current_skills=profile.current_skills,
        branch=profile.branch,
    )
    pred = ml_manager.salary_model.predict(df)
    return float(pred[0])


def _mock_salary(profile: "UserProfile") -> float:
    """Simple formula that mirrors the notebook's feature importances."""
    base = 5.0
    base += (profile.cgpa - 6.0) * 1.4
    base += profile.internships * 0.85
    skills = profile.current_skills
    if "ML" in skills:       base += 2.1
    if "DSA" in skills:      base += 1.4
    if "Cloud" in skills:    base += 1.3
    if "Python" in skills:   base += 0.9
    if "DevOps" in skills:   base += 0.7
    if "Java" in skills:     base += 0.6
    if "SQL" in skills:      base += 0.5
    if "Web_Dev" in skills:  base += 0.4
    base += profile.projects * 0.35
    base += profile.hackathons * 0.25
    base += profile.certifications * 0.30
    base -= profile.backlogs * 0.4
    if profile.branch.upper() in ("CSE", "IT"):
        base += 0.5
    return round(max(3.5, min(25.0, base)), 2)


def _build_importance_hints(profile: "UserProfile", predicted: float) -> list[FeatureImportanceHint]:
    skills = profile.current_skills
    return [
        FeatureImportanceHint(feature="CGPA",            importance=0.233, your_value=profile.cgpa),
        FeatureImportanceHint(feature="Internships",     importance=0.178, your_value=profile.internships),
        FeatureImportanceHint(feature="ML (skill)",      importance=0.119, your_value=int("ML" in skills)),
        FeatureImportanceHint(feature="DSA (skill)",     importance=0.082, your_value=int("DSA" in skills)),
        FeatureImportanceHint(feature="Projects",        importance=0.070, your_value=profile.projects),
        FeatureImportanceHint(feature="Cloud (skill)",   importance=0.068, your_value=int("Cloud" in skills)),
        FeatureImportanceHint(feature="Certifications",  importance=0.063, your_value=profile.certifications),
        FeatureImportanceHint(feature="Hackathons",      importance=0.048, your_value=profile.hackathons),
        FeatureImportanceHint(feature="Python (skill)",  importance=0.040, your_value=int("Python" in skills)),
    ]


# ── Public API ────────────────────────────────────────────────────────────────


def predict_salary(profile: "UserProfile") -> SalaryPredictionResponse:
    """
    Run the salary model (or fall back to mock) and return a
    SalaryPredictionResponse with feature-importance hints.
    """
    is_mock = not ml_manager.models_loaded

    if is_mock:
        lpa = _mock_salary(profile)
    else:
        try:
            lpa = _run_salary_model(profile)
        except Exception as exc:
            logger.warning("Salary model predict() failed: %s — using mock", exc)
            lpa = _mock_salary(profile)
            is_mock = True

    lpa = round(lpa, 2)
    return SalaryPredictionResponse(
        predicted_lpa=lpa,
        range_low=round(max(0.0, lpa - 0.88), 1),
        range_high=round(lpa + 0.88, 1),
        feature_importance_hints=_build_importance_hints(profile, lpa),
        mock=is_mock,
    )


def simulate_boost(profile: "UserProfile") -> SkillBoostResponse:
    """
    The Magic Feature™
    For every top-industry skill the user does NOT have, create a clone
    of their profile with that skill flipped to 1, predict salary, and
    return the delta sorted by highest impact.
    """
    is_mock = not ml_manager.models_loaded

    # Baseline
    if is_mock:
        baseline = _mock_salary(profile)
        predict_fn = _mock_salary
    else:
        try:
            baseline = _run_salary_model(profile)
        except Exception as exc:
            logger.warning("Boost: baseline failed (%s) — using mock", exc)
            baseline = _mock_salary(profile)
            is_mock = True

        if is_mock:
            predict_fn = _mock_salary
        else:
            predict_fn = _run_salary_model  # type: ignore[assignment]

    baseline = round(baseline, 2)
    boosts: list[BoostEntry] = []

    for skill in BOOST_CANDIDATE_SKILLS:
        if skill not in profile.current_skills:
            # Build a mutated profile  (we need a copy without Pydantic validation overhead)
            mutated_skills = profile.current_skills + [skill]

            # Reuse profile but swap skills
            from app.models.schemas import UserProfile as _UP  # local import avoids circulars
            mutated = _UP(
                cgpa=profile.cgpa,
                year=profile.year,
                branch=profile.branch,
                backlogs=profile.backlogs,
                internships=profile.internships,
                projects=profile.projects,
                hackathons=profile.hackathons,
                certifications=profile.certifications,
                current_skills=mutated_skills,
            )

            try:
                new_lpa = predict_fn(mutated)
            except Exception:
                new_lpa = _mock_salary(mutated)

            delta = round(new_lpa - baseline, 2)
            if delta > 0.01:
                boosts.append(BoostEntry(skill=skill, boost_lpa=delta))

    # Sort descending by impact
    boosts.sort(key=lambda b: b.boost_lpa, reverse=True)

    return SkillBoostResponse(
        current_lpa=baseline,
        boost_simulations=boosts,
        mock=is_mock,
    )
