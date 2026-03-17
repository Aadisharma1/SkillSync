"""
app/services/analyzer.py
-------------------------
Skill-gap analysis and demand forecasting.

Gap model:
  - MultiOutputClassifier(RandomForestClassifier)
  - Input:  one-hot Target_Role (10) + 55 binary skill bits = 65 cols
  - Output: 55 binary `miss_<skill>` targets
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.core.config import ALL_SKILLS_SORTED, VALID_ROLES
from app.core.ml_manager import ml_manager
from app.models.schemas import (
    DemandForecastResponse,
    DemandTrend,
    LearningStep,
    SkillGapResponse,
)

if TYPE_CHECKING:
    from app.models.schemas import UserProfile

logger = logging.getLogger("skillsync.analyzer")


# ══════════════════════════════════════════════════════════════════════════════
#  Learning tips  (fallback text for roadmap items)
# ══════════════════════════════════════════════════════════════════════════════

_LEARNING_TIPS: dict[str, str] = {
    "Python": "Start with freeCodeCamp Python or CS50P — 4 weeks to proficiency",
    "SQL": "Practice on SQLZoo, then HackerRank SQL — 2 weeks solid",
    "Machine Learning": "Andrew Ng's ML Specialisation on Coursera is the gold standard",
    "Docker": "Docker official 'Get Started' guide + build one containerised project",
    "Kubernetes": "KubeByExample.com → then deploy a 3-tier app on Minikube",
    "AWS": "AWS free-tier + Cloud Practitioner cert roadmap (60 hours)",
    "React": "Official React docs + build a full CRUD app with hooks",
    "Git": "learngitbranching.js.org game — interactive and fast",
    "Linux": "practise shells on OverTheWire Bandit (free, online)",
    "TypeScript": "TypeScript Handbook at typescriptlang.org — 1 week",
    "TensorFlow": "TensorFlow official tutorials; start with Keras Sequential API",
    "PyTorch": "PyTorch.org 60-min blitz + implement a CNN from scratch",
    "MLOps": "MLflow quickstart + build a simple model-serving pipeline",
    "REST APIs": "Build a CRUD API with FastAPI + deploy to Railway",
    "System Design": "System Design Primer on GitHub + design 3 real systems",
    "Deep Learning": "fast.ai Practical Deep Learning course (free)",
    "Pandas": "Kaggle 'Pandas' micro-course — 4 hours, hands-on",
    "NumPy": "NumPy quickstart tutorial, then Numpy 100 exercises repo",
    "Statistics": "Khan Academy Statistics + StatQuest YouTube channel",
    "Data Visualization": "Matplotlib + Seaborn tutorial; Plotly for interactive charts",
    "Scikit-learn": "Scikit-learn user guide + practice on Kaggle competitions",
    "Feature Engineering": "Kaggle 'Feature Engineering' course; read 'Feature Engineering for ML'",
    "Model Deployment": "FastAPI + Docker + Railway free-tier deployment walkthrough",
    "LLMs": "Andrej Karpathy's 'Build a GPT from scratch' on YouTube",
    "Ansible": "Ansible official Getting Started + automate local VM provisioning",
    "Authentication": "Implement JWT auth in FastAPI or Express — 1-week project",
    "Azure": "Microsoft Learn AZ-900 free path (about 30 hours)",
    "CI/CD": "GitHub Actions official docs + set up a pipeline for your repo",
    "CSS": "Kevin Powell's CSS course on YouTube — free and comprehensive",
    "Communication": "Toastmasters or workplace practice; 'Talk Like TED' book",
    "Cryptography": "Christof Paar's 'Introduction to Cryptography' on YouTube",
    "DSA": "NeetCode 150 roadmap + LeetCode Easy/Medium daily habit",
    "Data Cleaning": "Kaggle 'Data Cleaning' course + work with messy Kaggle datasets",
    "Ethical Hacking": "TryHackMe free paths: 'Pre-Security' then 'Jr Penetration Tester'",
    "Excel": "Microsoft Excel official training for Power users",
    "Firewalls": "pfSense documentation + set up a home lab firewall",
    "HTML": "MDN HTML tutorial — 1 week to build solid page-structure skills",
    "Incident Response": "SANS SEC504 sample material; practice in TryHackMe SOC Level 1",
    "Java": "Official Java Tutorial + Effective Java book",
    "JavaScript": "javascript.info — the most comprehensive free JS tutorial",
    "Monitoring": "Prometheus + Grafana getting-started guide; monitor a local app",
    "Networking": "Professor Messer's CompTIA Network+ free video course",
    "OOP": "Head First Object-Oriented Analysis & Design — practical approach",
    "Power BI": "Microsoft Learn Power BI free learning path (8 hours)",
    "Redis": "Redis University free courses at university.redis.com",
    "Responsive Design": "CSS Grid and Flexbox Froggy games + build 3 responsive layouts",
    "Risk Assessment": "NIST Risk Management Framework overview (free PDF)",
    "SIEM Tools": "ELK Stack tutorial + Splunk free training",
    "Security": "CompTIA Security+ study guide; full free path on TryHackMe",
    "Shell Scripting": "Linux Command Line book by William Shotts (free online)",
    "Tableau": "Tableau free eLearning + build 3 interactive dashboards",
    "Terraform": "HashiCorp Learn: Terraform Getting Started (free, ~6 hours)",
    "Testing": "Google Testing Blog + Jest or pytest fundamentals",
    "Unit Testing": "pytest documentation + TDD with Python (book)",
    "Webpack": "Webpack official guides + SurviveJS Webpack book",
}


def _get_tip(skill: str) -> str:
    return _LEARNING_TIPS.get(
        skill,
        "Search for beginner tutorials on Coursera, YouTube, or official docs",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Demand forecast static data
#  (derived from skill_demand_forecasting.ipynb)
# ══════════════════════════════════════════════════════════════════════════════

_DEMAND_DATA: list[dict] = [
    {
        "skill": "Python",
        "history": [1220,1218,1272,1330,1282,1310,1345,1320,1380,1400,1430,1410,
                    1450,1465,1488,1510,1530,1555,1570,1590,1610,1640,1655,1680,
                    1700,1725,1740,1762,1780,1805,1820,1840,1860,1885,1900,1920],
        "growth_pct": 4.36,
        "color": "#52b8d9",
    },
    {
        "skill": "Cloud",
        "history": [980,990,1010,1025,1015,1030,1055,1060,1080,1095,1110,1120,
                    1140,1150,1165,1180,1195,1205,1220,1235,1250,1260,1275,1290,
                    1300,1315,1330,1345,1360,1372,1385,1398,1410,1425,1438,1450],
        "growth_pct": 3.79,
        "color": "#52d9a4",
    },
    {
        "skill": "Machine Learning",
        "history": [760,770,775,790,800,810,822,835,848,860,875,885,900,912,925,
                    940,952,968,980,994,1008,1022,1038,1052,1068,1082,1098,1115,
                    1130,1145,1162,1178,1195,1212,1228,1245],
        "growth_pct": 4.98,
        "color": "#f07aff",
    },
    {
        "skill": "Data Science",
        "history": [650,660,655,670,668,672,675,680,678,682,685,688,686,690,692,
                    695,693,697,699,702,700,704,706,709,707,711,713,716,714,718,
                    720,723,721,725,727,730],
        "growth_pct": -0.09,
        "color": "#f7c86e",
    },
    {
        "skill": "Web Development",
        "history": [1100,1090,1080,1075,1060,1050,1040,1035,1020,1015,1005,998,
                    990,985,975,970,962,955,948,940,935,928,920,914,908,902,895,
                    888,882,876,870,864,858,852,846,840],
        "growth_pct": -16.96,
        "color": "#e8625a",
    },
    {
        "skill": "DevOps",
        "history": [580,588,595,605,612,620,630,638,648,656,665,674,682,692,700,
                    710,720,730,741,751,762,773,784,795,807,818,830,842,854,866,
                    879,891,904,917,930,943],
        "growth_pct": 5.82,
        "color": "#e8ac5a",
    },
    {
        "skill": "GenAI",
        "history": [200,215,230,248,265,285,308,332,358,385,415,447,481,518,556,
                    597,641,688,738,791,848,908,972,1040,1112,1188,1268,1353,1443,
                    1538,1638,1744,1856,1974,2099,2232],
        "growth_pct": 9.31,
        "color": "#c4bdff",
    },
]


# ══════════════════════════════════════════════════════════════════════════════
#  Public functions
# ══════════════════════════════════════════════════════════════════════════════


def analyze_gap(profile: "UserProfile") -> SkillGapResponse:
    """
    Compare the user's skills against the target role requirements.

    Uses the MultiOutputClassifier when available; falls back to rule-based
    set-difference logic (which is exactly what the model was trained to do).
    """
    if not profile.target_role:
        raise ValueError("target_role is required for gap analysis")

    target_role = profile.target_role
    if target_role not in VALID_ROLES:
        raise ValueError(
            f"Unknown role '{target_role}'. Valid roles: {', '.join(VALID_ROLES)}"
        )

    # Get the canonical required skills from metadata (or fallback dict)
    meta = ml_manager.skill_gap_meta
    role_skills: dict[str, list[str]] = meta.get("role_skills", _FALLBACK_ROLE_SKILLS)
    required: list[str] = role_skills.get(target_role, [])
    user_skills: list[str] = profile.current_skills

    is_mock = not ml_manager.models_loaded

    if not is_mock:
        # ── Live model path ───────────────────────────────────────────────
        try:
            df = ml_manager.build_gap_feature_vector(
                current_skills=user_skills,
                target_role=target_role,
            )
            # predict_proba returns list of (n_samples, n_classes) arrays
            # We take probability of class=1 (=missing) for each skill
            probas = ml_manager.skill_gap_model.predict_proba(df)
            # probas is list of 55 arrays shape (1, 2)
            miss_probs = [float(p[0][1]) for p in probas]
            # Threshold 0.5 for binary miss/have
            predicted_missing = [
                ALL_SKILLS_SORTED[i]
                for i, prob in enumerate(miss_probs)
                if prob >= 0.5 and ALL_SKILLS_SORTED[i] in required
            ]
            if not predicted_missing:
                # Model says nothing is missing — trust set-diff as sanity check
                predicted_missing = [s for s in required if s not in user_skills]
        except Exception as exc:
            logger.warning("Gap model predict failed: %s — using set-diff", exc)
            predicted_missing = [s for s in required if s not in user_skills]
            is_mock = True
    else:
        # ── Mock / fallback: pure set-difference ─────────────────────────
        predicted_missing = [s for s in required if s not in user_skills]

    skills_you_have = [s for s in required if s in user_skills]
    bonus_skills = [s for s in user_skills if s not in required]
    suitability = round(
        (len(skills_you_have) / len(required) * 100) if required else 100.0, 1
    )

    roadmap = [
        LearningStep(skill=s, tip=_get_tip(s)) for s in predicted_missing[:8]
    ]

    return SkillGapResponse(
        target_role=target_role,
        suitability_score=suitability,
        skills_you_have=skills_you_have,
        missing_skills=predicted_missing,
        bonus_skills=bonus_skills,
        learning_roadmap=roadmap,
        mock=is_mock,
    )


def forecast_demand() -> DemandForecastResponse:
    """
    Return structured time-series data for skill demand charts.
    Data is derived from skill_demand_forecasting.ipynb (LinearRegression
    on monthly job-posting counts Jan 2022 – Jun 2026 forecast).
    """
    trends = [DemandTrend(**entry) for entry in _DEMAND_DATA]
    return DemandForecastResponse(trends=trends, mock=False)


# ══════════════════════════════════════════════════════════════════════════════
#  Fallback role→skills mapping  (mirrors skill_gap_meta.json)
# ══════════════════════════════════════════════════════════════════════════════

_FALLBACK_ROLE_SKILLS: dict[str, list[str]] = {
    "Data Scientist":       ["Python","SQL","Machine Learning","Pandas","NumPy","Statistics",
                             "Data Visualization","Scikit-learn","Model Deployment","Feature Engineering"],
    "Software Engineer":    ["Java","DSA","OOP","Git","System Design","REST APIs",
                             "Unit Testing","SQL","Linux","Docker"],
    "Cloud Engineer":       ["Linux","AWS","Docker","Kubernetes","Networking","Terraform",
                             "CI/CD","Shell Scripting","Azure","Security"],
    "Data Analyst":         ["Excel","SQL","Power BI","Python","Statistics","Data Cleaning",
                             "Tableau","Communication","Pandas","Data Visualization"],
    "AI Engineer":          ["Python","Machine Learning","Deep Learning","Pandas",
                             "Model Deployment","PyTorch","TensorFlow","LLMs","MLOps",
                             "Feature Engineering"],
    "Frontend Developer":   ["HTML","CSS","JavaScript","React","TypeScript","Git",
                             "REST APIs","Responsive Design","Testing","Webpack"],
    "Backend Developer":    ["Python","Java","SQL","REST APIs","Docker","System Design",
                             "Authentication","Git","Redis","Linux"],
    "DevOps Engineer":      ["Linux","Docker","Kubernetes","CI/CD","Git","AWS",
                             "Monitoring","Shell Scripting","Ansible","Terraform"],
    "Cybersecurity Analyst":["Networking","Linux","Security","Ethical Hacking","Cryptography",
                             "SIEM Tools","Risk Assessment","Python","Incident Response","Firewalls"],
    "ML Engineer":          ["Python","Machine Learning","Scikit-learn","MLOps","Docker",
                             "Feature Engineering","SQL","Model Deployment","PyTorch","TensorFlow"],
}
