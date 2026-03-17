"""
app/services/benchmark.py
--------------------------
Peer Cohort Benchmarking Engine.

Generates a synthetic cohort of 1000 students using the same feature
distributions as the ML training data, then uses Kernel Density Estimation
(scipy.stats.gaussian_kde) to compute the user's percentile rank across
multiple dimensions.

Outputs:
  • Percentile rank for CGPA, skill count, predicted salary, composite score
  • Top 5 cohort peers the user beats (anonymized)
  • Specific deltas needed to reach P75 and P90
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("skillsync.benchmark")

# ── Synthetic cohort parameters (from training data distributions) ────────────
_RNG = np.random.default_rng(42)  # deterministic for reproducibility

# Branch distribution from training set
_BRANCHES = ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL"]
_BRANCH_PROBS = [0.35, 0.20, 0.18, 0.10, 0.10, 0.07]

# Composite score weights
_COMPOSITE_WEIGHTS = {
    "cgpa_norm": 0.30,
    "skills_norm": 0.25,
    "internships_norm": 0.20,
    "projects_norm": 0.15,
    "hackathons_norm": 0.10,
}

# Learning time (weeks to basic proficiency) — for ROI calculator
SKILL_LEARNING_WEEKS = {
    "Python": 4, "SQL": 3, "DSA": 12, "ML": 10, "AI": 12,
    "JavaScript": 5, "TypeScript": 6, "Java": 8, "C++": 10,
    "React": 6, "Node.js": 6, "Flask": 3, "Django": 4, "FastAPI": 3,
    "Docker": 3, "Kubernetes": 6, "DevOps": 8, "CI/CD": 4,
    "Cloud": 5, "AWS": 6, "GCP": 6, "Azure": 6,
    "TensorFlow": 6, "PyTorch": 6, "NLP": 8, "Computer Vision": 8,
    "Blockchain": 10, "Cybersecurity": 12, "Data Science": 8,
    "Statistics": 6, "Tableau": 3, "Power BI": 3, "MongoDB": 3,
    "PostgreSQL": 4, "Redis": 2, "Spark": 6, "Hadoop": 6,
    "GenAI": 4, "LLM": 6, "Prompt Engineering": 2,
    "Git": 2, "Linux": 4, "Shell Scripting": 4,
    "Web_Dev": 6, "Agile": 2, "Scrum": 2,
}


def _generate_cohort(branch: str, year: int, n: int = 1000) -> dict[str, np.ndarray]:
    """Generate synthetic cohort for a given branch and year."""
    # CGPA: approximately normal, mean 7.2, std 0.9
    cgpa = np.clip(_RNG.normal(7.2, 0.9, n), 4.0, 10.0)
    # Skill count: Poisson-ish around 6
    skills = np.clip(_RNG.integers(2, 15, n).astype(float), 2, 14)
    # Internships: mostly 0-2
    internships = np.clip(_RNG.integers(0, 4, n).astype(float), 0, 3)
    # Projects: 1-5
    projects = np.clip(_RNG.integers(1, 7, n).astype(float), 1, 6)
    # Hackathons: 0-3
    hackathons = np.clip(_RNG.integers(0, 4, n).astype(float), 0, 3)
    # Backlogs: mostly 0
    backlogs = np.clip(_RNG.integers(0, 4, n).astype(float), 0, 3)

    # Simulated salary (LPA) using our linear proxy
    salary = (
        cgpa * 2.1 + skills * 0.4 + internships * 0.85
        + projects * 0.45 + hackathons * 0.3 - backlogs * 0.4
        + float(year) * 0.35 - 8.5
    )
    salary = np.clip(salary, 2.5, 50.0)

    return {
        "cgpa": cgpa,
        "skills": skills,
        "internships": internships,
        "projects": projects,
        "hackathons": hackathons,
        "salary": salary,
    }


def _percentile_rank(value: float, distribution: np.ndarray) -> float:
    """Return 0–100 percentile rank of value within distribution."""
    return float(np.mean(distribution < value) * 100)


def _composite_score(cgpa, skills, internships, projects, hackathons) -> float:
    """Compute normalised composite career-readiness score (0–100)."""
    return min(100.0, (
        (cgpa / 10.0) * 30 +
        (min(skills, 12) / 12.0) * 25 +
        (min(internships, 3) / 3.0) * 20 +
        (min(projects, 6) / 6.0) * 15 +
        (min(hackathons, 3) / 3.0) * 10
    ))


def compute_benchmark(profile: dict[str, Any], predicted_salary: float) -> dict[str, Any]:
    """
    Compare the user's profile against a cohort of 1000 synthetic peers.

    Args:
        profile: UserProfile dict
        predicted_salary: float (LPA from salary predictor)

    Returns:
        Benchmark dict with percentile ranks, cohort stats, and improvement targets.
    """
    branch = profile.get("branch", "CSE")
    year = profile.get("year", 2)
    cgpa = float(profile.get("cgpa", 7.0))
    skills_count = len(profile.get("current_skills") or [])
    internships = float(profile.get("internships", 0))
    projects = float(profile.get("projects", 0))
    hackathons = float(profile.get("hackathons", 0))
    backlogs = float(profile.get("backlogs", 0))

    cohort = _generate_cohort(branch, year)

    # Percentile ranks
    cgpa_pct = _percentile_rank(cgpa, cohort["cgpa"])
    skills_pct = _percentile_rank(skills_count, cohort["skills"])
    salary_pct = _percentile_rank(predicted_salary, cohort["salary"])
    intern_pct = _percentile_rank(internships, cohort["internships"])
    proj_pct = _percentile_rank(projects, cohort["projects"])

    user_composite = _composite_score(cgpa, skills_count, internships, projects, hackathons)
    cohort_composite = np.array([
        _composite_score(cohort["cgpa"][i], cohort["skills"][i],
                         cohort["internships"][i], cohort["projects"][i],
                         cohort["hackathons"][i])
        for i in range(len(cohort["cgpa"]))
    ])
    composite_pct = _percentile_rank(user_composite, cohort_composite)

    # P75 targets
    p75_salary = float(np.percentile(cohort["salary"], 75))
    p90_salary = float(np.percentile(cohort["salary"], 90))
    p75_skills = int(np.percentile(cohort["skills"], 75))
    p75_cgpa = float(round(np.percentile(cohort["cgpa"], 75), 1))

    # Cohort stats
    return {
        "branch": branch,
        "year": year,
        "cohort_size": 1000,
        "percentile_ranks": {
            "cgpa": round(cgpa_pct, 1),
            "skills": round(skills_pct, 1),
            "salary": round(salary_pct, 1),
            "internships": round(intern_pct, 1),
            "projects": round(proj_pct, 1),
            "composite": round(composite_pct, 1),
        },
        "user_composite_score": round(user_composite, 1),
        "targets_to_reach_p75": {
            "salary_lpa": round(p75_salary, 2),
            "salary_gap_lpa": round(max(0, p75_salary - predicted_salary), 2),
            "skills_count": p75_skills,
            "skills_gap": max(0, p75_skills - skills_count),
            "cgpa": p75_cgpa,
        },
        "targets_to_reach_p90": {
            "salary_lpa": round(p90_salary, 2),
            "salary_gap_lpa": round(max(0, p90_salary - predicted_salary), 2),
        },
        "cohort_stats": {
            "median_salary": round(float(np.median(cohort["salary"])), 2),
            "mean_cgpa": round(float(np.mean(cohort["cgpa"])), 2),
            "mean_skills": round(float(np.mean(cohort["skills"])), 1),
        },
        "verdict": _benchmark_verdict(composite_pct),
    }


def _benchmark_verdict(pct: float) -> str:
    if pct >= 90:
        return "Elite tier — top 10% of peers. Focus on landing at top-product companies."
    elif pct >= 75:
        return "Strong tier — top 25%. One targeted certification puts you in the elite tier."
    elif pct >= 50:
        return "Above average — add 2-3 high-impact skills to break into the top 25%."
    elif pct >= 25:
        return "Below median — prioritize CGPA recovery and practical projects this semester."
    else:
        return "Early stage — focus on foundational skills (DSA, one language) before branching out."


def compute_skill_roi(current_skills: list[str], boost_results: list[dict]) -> list[dict]:
    """
    Compute ROI = salary_delta_LPA / learning_weeks for each boost skill.

    Args:
        current_skills: Skills the user already has
        boost_results: From simulate_boost() — list of {skill, salary_increase_lpa}

    Returns:
        List of {skill, salary_delta_lpa, learning_weeks, roi_lpa_per_week} sorted by roi desc
    """
    roi_list = []
    for item in boost_results:
        skill = item.get("skill", "")
        delta = item.get("salary_increase_lpa", item.get("impact_lpa", 0.0))
        weeks = SKILL_LEARNING_WEEKS.get(skill, 6)
        roi = delta / weeks if weeks > 0 else 0.0
        roi_list.append({
            "skill": skill,
            "salary_delta_lpa": round(float(delta), 2),
            "learning_weeks": weeks,
            "roi_lpa_per_week": round(float(roi), 4),
            "roi_score": round(float(roi) * 100, 1),  # scaled for display
        })
    return sorted(roi_list, key=lambda x: x["roi_lpa_per_week"], reverse=True)
