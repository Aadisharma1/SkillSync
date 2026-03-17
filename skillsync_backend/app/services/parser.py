"""
app/services/parser.py
----------------------
Zero-Click Auto-Onboarding: AI-powered resume parser.

Pipeline:
  1. PyMuPDF extracts raw text from every page of the uploaded PDF in <50ms.
  2. Text is sent to Groq's LLM (llama-3.3-70b) with a strict system prompt
     that enforces exact JSON output matching our UserProfile schema.
  3. A robust extraction block handles malformed LLM output gracefully.
  4. Falls back to a regex-based heuristic parser if LLM key is missing.

GROQ_API_KEY must be set in the environment (or a .env file).
Get a free key at https://console.groq.com (no credit card needed).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger("skillsync.parser")

# ── All skills the model knows about ─────────────────────────────────────────
_ALL_SKILLS = [
    "Python", "Java", "C++", "JavaScript", "TypeScript", "SQL", "R",
    "DSA", "ML", "AI", "DevOps", "Cloud", "AWS", "GCP", "Azure",
    "Docker", "Kubernetes", "CI/CD", "Linux", "Shell Scripting",
    "Web_Dev", "React", "Node.js", "Flask", "Django", "FastAPI",
    "TensorFlow", "PyTorch", "NLP", "Computer Vision", "Blockchain",
    "Cybersecurity", "Data Science", "Statistics", "Tableau", "Power BI",
    "MongoDB", "PostgreSQL", "Redis", "Spark", "Hadoop", "GenAI",
    "LLM", "Prompt Engineering", "Git", "Agile", "Scrum",
]

_VALID_BRANCHES = ["CSE", "IT", "ECE", "EEE", "MECH", "CIVIL", "MBA", "Other"]

# ── System prompt — enforces strict JSON matching our schema ──────────────────
_SYSTEM_PROMPT = f"""You are a resume information extractor. Your ONLY job is to extract
structured data from resume text and return a SINGLE valid JSON object.

The JSON MUST match this schema exactly (no extra keys, no missing keys):
{{
  "cgpa": <float, 0.0–10.0, extract from GPA/CGPA/percentage; if percentage convert /10>,
  "year": <int 1-4, current academic year; infer from graduation year if needed>,
  "branch": <string, ONE of: {_VALID_BRANCHES}; map closest match>,
  "backlogs": <int, number of backlogs/failed subjects; default 0 if not mentioned>,
  "internships": <int, number of internships completed>,
  "projects": <int, number of projects listed>,
  "hackathons": <int, number of hackathons participated in>,
  "certifications": <int, number of certifications listed>,
  "current_skills": <list[string], ONLY include skills from this exact list: {_ALL_SKILLS}>,
  "target_role": null
}}

CRITICAL RULES:
1. Return ONLY the JSON object — no explanation, no markdown, no code blocks.
2. current_skills must only contain skills from the provided list, verbatim.
3. If a field cannot be determined, use the most reasonable default (0 for ints, 7.0 for cgpa, 2 for year).
4. Never include keys not in the schema.
5. cgpa must be a float between 0.0 and 10.0."""


def _extract_json_from_text(text: str) -> dict[str, Any]:
    """
    Robustly parse JSON from LLM output.
    Handles: bare JSON, JSON in code blocks, JSON with surrounding text.
    """
    # 1. Try direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code blocks ```json ... ``` or ``` ... ```
    code_block = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1))
        except json.JSONDecodeError:
            pass

    # 3. Find the first {...} block in the text
    brace_match = re.search(r"\{[\s\S]+\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"LLM did not return valid JSON. Raw response:\n{text[:500]}")


def _regex_fallback(text: str) -> dict[str, Any]:
    """
    Heuristic regex parser — used when GROQ_API_KEY is absent.
    Extracts skills and basic metrics from resume text without an LLM.
    """
    text_lower = text.lower()

    # CGPA / GPA
    cgpa = 7.0
    cgpa_match = re.search(r"(?:cgpa|gpa|grade)[:\s]+(\d+\.?\d*)", text_lower)
    if cgpa_match:
        val = float(cgpa_match.group(1))
        cgpa = val if val <= 10 else round(val / 10, 2)

    # Branch
    branch = "CSE"
    branch_map = {
        "computer science": "CSE", "cse": "CSE", "information technology": "IT",
        " it ": "IT", "electronics": "ECE", "electrical": "EEE",
        "mechanical": "MECH", "civil": "CIVIL", "mba": "MBA",
    }
    for kw, b in branch_map.items():
        if kw in text_lower:
            branch = b
            break

    # Year
    year = 2
    year_match = re.search(r"(\d)(?:st|nd|rd|th)\s+year", text_lower)
    if year_match:
        year = min(4, max(1, int(year_match.group(1))))

    # Counts
    internships = len(re.findall(r"\bintern\b", text_lower))
    projects = len(re.findall(r"\bproject\b", text_lower))
    hackathons = len(re.findall(r"\bhackathon\b", text_lower))
    certifications = len(re.findall(r"\bcertif", text_lower))

    # Skills — match exact strings from ALL_SKILLS list (case-insensitive)
    found_skills = []
    for skill in _ALL_SKILLS:
        pattern = r"\b" + re.escape(skill.lower().replace("_", r"[\s_]?")) + r"\b"
        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return {
        "cgpa": cgpa,
        "year": year,
        "branch": branch,
        "backlogs": 0,
        "internships": min(internships, 5),
        "projects": min(projects, 10),
        "hackathons": min(hackathons, 5),
        "certifications": min(certifications, 5),
        "current_skills": found_skills,
        "target_role": None,
    }


def extract_profile_from_pdf(pdf_bytes: bytes) -> dict[str, Any]:
    """
    Extract a structured UserProfile dict from raw PDF bytes.

    Steps:
      1. PyMuPDF extracts full text from all pages.
      2. Groq LLM (llama-3.3-70b-versatile) converts text → JSON.
      3. Regex fallback if GROQ_API_KEY not set.

    Returns:
        dict matching the UserProfile schema.

    Raises:
        ValueError: If PDF is empty or text extraction fails.
    """
    # Step 1 — Extract text with PyMuPDF
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise RuntimeError(
            "PyMuPDF is not installed. Run: pip install PyMuPDF"
        )

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text("text"))
    doc.close()

    raw_text = "\n".join(pages_text).strip()
    if not raw_text:
        raise ValueError("The uploaded PDF appears to be empty or image-only (no extractable text).")

    # Truncate to ~6000 chars to fit LLM context window
    resume_text = raw_text[:6000]
    logger.info("Extracted %d chars from PDF, truncated to %d", len(raw_text), len(resume_text))

    # Step 2 — LLM extraction via Groq
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        logger.warning("GROQ_API_KEY not set — using regex fallback parser")
        return _regex_fallback(resume_text)

    try:
        from groq import Groq  # type: ignore
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract profile from this resume:\n\n{resume_text}"},
            ],
            temperature=0.0,   # deterministic
            max_tokens=512,
        )

        raw_output = response.choices[0].message.content or ""
        logger.debug("LLM raw output: %s", raw_output[:300])
        profile = _extract_json_from_text(raw_output)

        # Validate + clamp
        profile["cgpa"] = max(0.0, min(10.0, float(profile.get("cgpa", 7.0))))
        profile["year"] = max(1, min(4, int(profile.get("year", 2))))
        if profile.get("branch") not in _VALID_BRANCHES:
            profile["branch"] = "CSE"
        profile["current_skills"] = [
            s for s in profile.get("current_skills", []) if s in _ALL_SKILLS
        ]
        profile.setdefault("target_role", None)
        return profile

    except Exception as exc:
        logger.exception("LLM extraction failed, falling back to regex: %s", exc)
        return _regex_fallback(resume_text)
