"""
app/core/config.py
------------------
Centralised paths and constants for SkillSync backend.
The ML artifacts live in the project root (next to the notebooks).
"""

import os
from pathlib import Path

# ── Project root = two levels up from this file ──────────────────────────────
# skillsync_backend/app/core/config.py  →  SkillSync/
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent

# ── ML Artifact Paths ─────────────────────────────────────────────────────────
ML_DIR = ROOT_DIR  # .pkl files sit directly in the project root

SALARY_MODEL_PATH = ML_DIR / "salary_model.pkl"
SKILL_GAP_MODEL_PATH = ML_DIR / "skill_gap_model.pkl"
FEATURE_COLUMNS_PATH = ML_DIR / "feature_columns.pkl"
SKILL_GAP_META_PATH = ML_DIR / "skill_gap_meta.json"

# ── Salary model feature constants ────────────────────────────────────────────
# Derived from salary_model.ipynb output:
# Index(['CGPA', 'Year', 'Backlogs', 'DSA', 'Python', 'Java', 'Web_Dev', 'ML',
#        'Cloud', 'DevOps', 'SQL', 'Internships', 'Projects', 'Hackathons',
#        'Certifications', 'Branch_ECE', 'Branch_IT', 'Branch_MECH'])
SALARY_SKILL_COLS = ["DSA", "Python", "Java", "Web_Dev", "ML", "Cloud", "DevOps", "SQL"]
SALARY_NUMERIC_COLS = ["CGPA", "Year", "Backlogs", "Internships", "Projects", "Hackathons", "Certifications"]
SALARY_BRANCH_COLS = ["Branch_ECE", "Branch_IT", "Branch_MECH"]

# Skills the boost simulator checks (existing model skills that the salary model knows about)
BOOST_CANDIDATE_SKILLS = ["DSA", "Python", "Java", "Web_Dev", "ML", "Cloud", "DevOps", "SQL"]

# ── Skill Gap model constants ──────────────────────────────────────────────────
# 55 skills from skill_gap_analyzer.ipynb, alphabetically sorted
ALL_SKILLS_SORTED = [
    "AWS", "Ansible", "Authentication", "Azure", "CI/CD", "CSS", "Communication",
    "Cryptography", "DSA", "Data Cleaning", "Data Visualization", "Deep Learning",
    "Docker", "Ethical Hacking", "Excel", "Feature Engineering", "Firewalls", "Git",
    "HTML", "Incident Response", "Java", "JavaScript", "Kubernetes", "LLMs", "Linux",
    "MLOps", "Machine Learning", "Model Deployment", "Monitoring", "Networking",
    "NumPy", "OOP", "Pandas", "Power BI", "PyTorch", "Python", "REST APIs", "React",
    "Redis", "Responsive Design", "Risk Assessment", "SIEM Tools", "SQL", "Scikit-learn",
    "Security", "Shell Scripting", "Statistics", "System Design", "Tableau", "TensorFlow",
    "Terraform", "Testing", "TypeScript", "Unit Testing", "Webpack",
]

VALID_ROLES = [
    "Data Scientist", "Software Engineer", "Cloud Engineer", "Data Analyst",
    "AI Engineer", "Frontend Developer", "Backend Developer", "DevOps Engineer",
    "Cybersecurity Analyst", "ML Engineer",
]

# One-hot columns that the gap model expects (Target_Role_<role> for each role + 55 skill bits)
GAP_ROLE_COLS = [f"Target_Role_{r}" for r in VALID_ROLES]
