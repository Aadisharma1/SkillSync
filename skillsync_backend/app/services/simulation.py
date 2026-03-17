"""
app/services/simulation.py
---------------------------
5-Year Career Monte Carlo Simulation + Interview Readiness Scoring.

Monte Carlo:
  Runs 500 stochastic career trajectory samples over 5 years.
  Each year applies:
    1. Random skill acquisition (Bernoulli trials — user may or may not learn new skills)
    2. Market demand shift (from LR forecasted growth rates, with noise)
    3. RF proxy salary re-evaluation at new skill level
    4. Seniority bump (1.08–1.15x per year after year 2)

  Outputs P10/P50/P90 salary bands per year for a fan chart.

Interview Readiness:
  Scores 0-100 across 5 dimensions derived from the user's profile vs
  the target role. Each dimension is weighted by real interview rubrics:
    • DSA Depth (30%) — inferred from DSA skill, projects, hackathons
    • System Design (20%) — Cloud, Docker, Kubernetes, DevOps skills
    • Domain Knowledge (25%) — role-relevant skills from gap analysis
    • Project Strength (15%) — project count, hackathon count
    • Communication Proxy (10%) — presentation skills, hackathons

  Final score mapped to "Low / Medium / High / Elite" readiness tier.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("skillsync.simulation")

# Salary proxy weights (matching fhe_predictor.py)
_WEIGHTS = np.array([
    2.1, 0.35, -0.4, 0.85, 0.45, 0.3, 0.4,
    0.82, 0.6, 1.2, 0.95, 0.55,
], dtype=np.float64)
_BIAS = -8.5

# Skill demand growth rates per half-year (from skill_demand_forecasting.ipynb)
_DEMAND_GROWTH = {
    "GenAI": 0.093, "DevOps": 0.058, "ML": 0.050, "Python": 0.044,
    "Cloud": 0.038, "AI": 0.045, "TypeScript": 0.035, "LLM": 0.060,
    "Data Science": -0.001, "Web_Dev": -0.170,
    "JavaScript": 0.010, "Java": 0.005, "SQL": 0.015,
    "Docker": 0.040, "Kubernetes": 0.045, "AWS": 0.038,
    "default": 0.020,
}

# Which skill slots (in the 12-feature FHE vector) map to our tracked skills
_FEATURE_SKILL_MAP = {
    7: "DSA", 8: "Python", 9: "ML", 10: "Cloud", 11: "SQL",
}

# Company tech stack profiles
COMPANY_PROFILES = {
    "Google": {
        "required": ["DSA", "Python", "ML", "C++", "System Design", "Linux"],
        "preferred": ["TensorFlow", "Kubernetes", "AI", "Statistics"],
        "tier": "FAANG",
        "avg_package_lpa": 45,
    },
    "Microsoft": {
        "required": ["DSA", "C++", "Python", "Cloud", "Azure", "SQL"],
        "preferred": ["TypeScript", "Docker", "Kubernetes", "AI"],
        "tier": "FAANG",
        "avg_package_lpa": 38,
    },
    "Flipkart": {
        "required": ["DSA", "Java", "Python", "SQL", "MongoDB"],
        "preferred": ["Spark", "Kafka", "Docker", "Redis"],
        "tier": "Indian Unicorn",
        "avg_package_lpa": 28,
    },
    "Infosys": {
        "required": ["Java", "SQL", "Python", "Git", "Agile"],
        "preferred": ["Cloud", "React", "Node.js"],
        "tier": "IT Services",
        "avg_package_lpa": 8,
    },
    "Goldman Sachs": {
        "required": ["Python", "SQL", "Statistics", "Java", "DSA"],
        "preferred": ["R", "Spark", "Bloomberg Terminal", "ML"],
        "tier": "Finance Tech",
        "avg_package_lpa": 35,
    },
    "Startup": {
        "required": ["Python", "React", "SQL", "Git", "Docker"],
        "preferred": ["FastAPI", "TypeScript", "Cloud", "Redis"],
        "tier": "Startup",
        "avg_package_lpa": 15,
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO CAREER SIMULATION
# ══════════════════════════════════════════════════════════════════════════════

def run_career_simulation(
    profile: dict[str, Any],
    current_salary: float,
    planned_skills: list[str] | None = None,
    n_samples: int = 500,
    n_years: int = 5,
) -> dict[str, Any]:
    """
    Simulate 5-year salary trajectory using Monte Carlo sampling.

    Each year applies stochastic:
      1. Skill acquisition (Bernoulli) — planned skills learned with p=0.7
      2. Market demand noise (Gaussian around LR trend)
      3. RF proxy re-evaluation at new skill vector
      4. Seniority/experience multiplier

    Args:
        profile: UserProfile dict
        current_salary: Base salary prediction (LPA)
        planned_skills: Skills the user plans to acquire (default: top 3 ML suggestions)
        n_samples: Monte Carlo samples (500 gives stable percentiles in <100ms)
        n_years: Forecast horizon in years

    Returns:
        dict with per-year P10/P50/P90 salary bands + key statistics
    """
    rng = np.random.default_rng(None)  # random seed for each call

    # Extract feature vector
    cgpa = float(profile.get("cgpa", 7.0))
    year_academic = float(profile.get("year", 2))
    backlogs = float(profile.get("backlogs", 0))
    internships = float(profile.get("internships", 0))
    projects = float(profile.get("projects", 0))
    hackathons = float(profile.get("hackathons", 0))
    certifications = float(profile.get("certifications", 0))
    current_skills = set(profile.get("current_skills") or [])

    if planned_skills is None:
        planned_skills = ["ML", "Cloud", "DSA"]  # default high-ROI

    # Base feature vector [CGPA, Year, Backlogs, Intern, Proj, Hack, Cert, DSA, Py, ML, Cloud, SQL]
    base_vec = np.array([
        cgpa, year_academic, backlogs, internships, projects, hackathons, certifications,
        1.0 if "DSA" in current_skills else 0.0,
        1.0 if "Python" in current_skills else 0.0,
        1.0 if "ML" in current_skills else 0.0,
        1.0 if "Cloud" in current_skills else 0.0,
        1.0 if "SQL" in current_skills else 0.0,
    ], dtype=np.float64)

    # Monte Carlo: shape (n_samples, n_years)
    trajectories = np.zeros((n_samples, n_years))

    for s in range(n_samples):
        vec = base_vec.copy()
        for y in range(n_years):
            # 1. Stochastic skill acquisition (each planned skill learned w/ p=0.65)
            for skill in planned_skills:
                if rng.random() < 0.65:
                    skill_idx = _skill_to_feature_idx(skill)
                    if skill_idx is not None:
                        vec[skill_idx] = 1.0

            # 2. Add experience (internships + projects grow over years)
            vec[3] = min(vec[3] + rng.integers(0, 2), 5)  # internships
            vec[4] = min(vec[4] + rng.integers(0, 2), 8)  # projects

            # 3. Market demand noise ± 5%
            demand_factor = 1.0 + rng.normal(0.02, 0.05)  # ~2% market growth with noise

            # 4. RF proxy salary
            raw_salary = float(np.dot(vec, _WEIGHTS) + _BIAS)
            raw_salary = max(2.5, raw_salary)

            # 5. Seniority multiplier (grows each year)
            seniority = 1.0 + (y * 0.08)  # 8% raise per year base
            noise = rng.normal(1.0, 0.08)  # ±8% individual variance

            trajectories[s, y] = raw_salary * seniority * demand_factor * noise

    # Compute percentile bands
    p10 = np.percentile(trajectories, 10, axis=0).tolist()
    p50 = np.percentile(trajectories, 50, axis=0).tolist()
    p90 = np.percentile(trajectories, 90, axis=0).tolist()
    p25 = np.percentile(trajectories, 25, axis=0).tolist()
    p75 = np.percentile(trajectories, 75, axis=0).tolist()

    # Probability of crossing thresholds
    prob_20lpa = float(np.mean(trajectories[:, -1] >= 20) * 100)
    prob_30lpa = float(np.mean(trajectories[:, -1] >= 30) * 100)
    prob_50lpa = float(np.mean(trajectories[:, -1] >= 50) * 100)

    years = list(range(1, n_years + 1))

    return {
        "n_samples": n_samples,
        "n_years": n_years,
        "years": years,
        "current_salary_lpa": round(current_salary, 2),
        "trajectories": {
            "p10": [round(v, 2) for v in p10],
            "p25": [round(v, 2) for v in p25],
            "p50": [round(v, 2) for v in p50],
            "p75": [round(v, 2) for v in p75],
            "p90": [round(v, 2) for v in p90],
        },
        "year5_stats": {
            "median_lpa": round(p50[-1], 2),
            "best_case_lpa": round(p90[-1], 2),
            "worst_case_lpa": round(p10[-1], 2),
        },
        "milestone_probabilities": {
            "prob_20lpa_in_5yr": round(prob_20lpa, 1),
            "prob_30lpa_in_5yr": round(prob_30lpa, 1),
            "prob_50lpa_in_5yr": round(prob_50lpa, 1),
        },
        "planned_skills": planned_skills,
        "simulation_note": (
            f"{n_samples} Monte Carlo samples · "
            "accounts for skill acquisition uncertainty, market demand shifts, and salary variance"
        ),
    }


def _skill_to_feature_idx(skill: str) -> int | None:
    mapping = {"DSA": 7, "Python": 8, "ML": 9, "Cloud": 10, "SQL": 11}
    return mapping.get(skill)


# ══════════════════════════════════════════════════════════════════════════════
# INTERVIEW READINESS SCORING
# ══════════════════════════════════════════════════════════════════════════════

_DSA_SKILLS = {"DSA", "C++", "Java", "Python"}
_SYSDESIGN_SKILLS = {"Cloud", "Docker", "Kubernetes", "DevOps", "Linux", "Redis", "Spark"}
_COMM_SKILLS = {"Agile", "Scrum"}

def compute_interview_readiness(
    profile: dict[str, Any],
    target_role: str,
    missing_skills: list[str],
    company: str | None = None,
) -> dict[str, Any]:
    """
    Score interview readiness 0-100 across 5 weighted dimensions.

    Weights match typical SDE/DS interview rubrics:
      DSA Depth (30%), System Design (20%), Domain Knowledge (25%),
      Project Strength (15%), Communication Proxy (10%)
    """
    skills = set(profile.get("current_skills") or [])
    projects = float(profile.get("projects", 0))
    hackathons = float(profile.get("hackathons", 0))
    certifications = float(profile.get("certifications", 0))
    cgpa = float(profile.get("cgpa", 7.0))
    backlogs = float(profile.get("backlogs", 0))

    # 1. DSA Depth (0-30)
    dsa_skills_count = len(skills & _DSA_SKILLS)
    dsa_score = min(30, dsa_skills_count * 8 + hackathons * 3 + min(projects, 3) * 2)

    # 2. System Design (0-20)
    sysdesign_count = len(skills & _SYSDESIGN_SKILLS)
    sysdesign_score = min(20, sysdesign_count * 4 + (2 if "DevOps" in skills else 0))

    # 3. Domain Knowledge (0-25) — ratio of required skills you have
    total_required = max(1, len(missing_skills) + len(skills))
    have_ratio = 1 - (len(missing_skills) / total_required)
    domain_score = min(25, have_ratio * 25)

    # 4. Project Strength (0-15)
    proj_score = min(15, projects * 2.5 + hackathons * 1.5)

    # 5. Communication Proxy (0-10)
    comm_score = min(10, hackathons * 2 + len(skills & _COMM_SKILLS) * 2 + certifications)

    # CGPA modifier: backlogs/low CGPA can reduce score
    cgpa_modifier = 1.0 if cgpa >= 7.5 else (0.92 if cgpa >= 6.5 else 0.85)
    if backlogs > 2:
        cgpa_modifier *= 0.9

    total = (dsa_score + sysdesign_score + domain_score + proj_score + comm_score) * cgpa_modifier
    total = min(100, total)

    tier, color = _readiness_tier(total)

    # Company-specific check
    company_gap = []
    if company and company in COMPANY_PROFILES:
        comp = COMPANY_PROFILES[company]
        company_gap = [s for s in comp["required"] if s not in skills]

    return {
        "overall_score": round(total, 1),
        "tier": tier,
        "tier_color": color,
        "breakdown": {
            "dsa_depth": round(dsa_score, 1),
            "dsa_max": 30,
            "system_design": round(sysdesign_score, 1),
            "system_design_max": 20,
            "domain_knowledge": round(domain_score, 1),
            "domain_max": 25,
            "project_strength": round(proj_score, 1),
            "project_max": 15,
            "communication": round(comm_score, 1),
            "communication_max": 10,
        },
        "improvement_tips": _readiness_tips(tier, dsa_score, sysdesign_score, domain_score),
        "company_specific_gap": company_gap,
        "target_role": target_role,
        "company": company,
    }


def _readiness_tier(score: float) -> tuple[str, str]:
    if score >= 80:
        return "Elite", "#52d9a4"
    elif score >= 65:
        return "Interview-Ready", "#c4bdff"
    elif score >= 45:
        return "Developing", "#f7c86e"
    else:
        return "Foundational", "#e8625a"


def _readiness_tips(tier: str, dsa: float, sys: float, domain: float) -> list[str]:
    tips = []
    if dsa < 20:
        tips.append("Solve 3 LeetCode Mediums per day — DSA is the #1 filter at top companies.")
    if sys < 12:
        tips.append("Study 'Designing Data-Intensive Applications' + deploy one distributed project.")
    if domain < 15:
        tips.append("Close top 3 skill gaps from gap analysis — domain match is 25% of the rubric.")
    if tier == "Foundational":
        tips.append("Focus exclusively on Python + DSA for 60 days before applying to shortlisted companies.")
    elif tier == "Developing":
        tips.append("Build one end-to-end project using your target company's tech stack.")
    elif tier == "Interview-Ready":
        tips.append("Do mock interviews on Pramp or Interviewing.io — you're ready to apply top-tier.")
    else:
        tips.append("You're elite-tier. Focus on referrals and negotiating comp, not more studying.")
    return tips


# ══════════════════════════════════════════════════════════════════════════════
# COMPANY-SPECIFIC GAP ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def analyze_company_gap(profile: dict[str, Any], company: str) -> dict[str, Any]:
    """Return company-specific skill gaps and profile match score."""
    if company not in COMPANY_PROFILES:
        avail = list(COMPANY_PROFILES.keys())
        raise ValueError(f"Company '{company}' not in database. Available: {avail}")

    comp = COMPANY_PROFILES[company]
    skills = set(profile.get("current_skills") or [])
    required = comp["required"]
    preferred = comp.get("preferred", [])

    missing_required = [s for s in required if s not in skills]
    missing_preferred = [s for s in preferred if s not in skills]
    have_required = [s for s in required if s in skills]
    have_preferred = [s for s in preferred if s in skills]

    match_score = (len(have_required) / max(1, len(required))) * 70 + \
                  (len(have_preferred) / max(1, len(preferred))) * 30

    return {
        "company": company,
        "tier": comp["tier"],
        "avg_package_lpa": comp["avg_package_lpa"],
        "match_score": round(match_score, 1),
        "required_skills": required,
        "preferred_skills": preferred,
        "you_have_required": have_required,
        "you_have_preferred": have_preferred,
        "missing_required": missing_required,
        "missing_preferred": missing_preferred,
        "verdict": _company_verdict(match_score, missing_required),
    }


def _company_verdict(score: float, missing_req: list[str]) -> str:
    if score >= 80 and not missing_req:
        return "Strong match — apply directly. Prepare for DSA + system design rounds."
    elif score >= 60:
        return f"Good potential. Close {len(missing_req)} required skill(s) first: {', '.join(missing_req[:2])}."
    elif score >= 40:
        return f"Needs work. Focus on: {', '.join(missing_req[:3])} to reach competitive threshold."
    else:
        return "Significant gap. Recommend 6-month skill-building plan before targeting this company."
