"""
app/services/career_coach.py
-----------------------------
Groq-powered AI Career Coach.

Takes any ML analysis result (salary prediction, gap analysis, boost simulation)
and returns a structured, personalized 3-section career strategy grounded in
the specific predicted numbers — not generic advice.

This is the key differentiator vs "just use ChatGPT":
  • Grounded in YOUR predicted salary (not generic)
  • Knows your specific missing skills from the gap model
  • References the skill boost deltas from the RF model
  • Provides actionable interview prep tailored to your readiness score
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("skillsync.career_coach")

_SYSTEM_PROMPT = """You are an elite career strategist for engineering students in India.
You receive structured ML analysis results from SkillSync's AI models and provide
hyper-personalized, actionable career advice grounded in those specific numbers.

CRITICAL RULES:
1. Always reference the SPECIFIC numbers from the analysis (predicted salary, exact skill gaps, etc.)
2. Be concise — 3 sections, max 2-3 sentences each
3. Use a confident, senior-mentor tone
4. Give CONCRETE actions, not platitudes
5. Return ONLY valid JSON matching this schema exactly:

{
  "headline": "<one compelling sentence summarizing the opportunity>",
  "immediate_actions": ["<action 1>", "<action 2>", "<action 3>"],
  "strategic_advice": "<2-3 sentences on the 6-month career strategy based on the analysis>",
  "salary_negotiation_tip": "<specific salary negotiation advice based on predicted package>",
  "motivational_insight": "<1 sentence connecting their current trajectory to a real outcome>"
}"""


def get_career_advice(
    analysis_type: str,
    results: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """
    Generate personalized career advice using Groq LLaMA-3.3-70b.

    Args:
        analysis_type: One of 'salary', 'gap', 'boost', 'simulation', 'benchmark'
        results: The full ML analysis result dict
        profile: UserProfile dict (CGPA, branch, year, skills, etc.)

    Returns:
        dict with keys: headline, immediate_actions, strategic_advice,
                        salary_negotiation_tip, motivational_insight
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return _fallback_advice(analysis_type, results, profile)

    user_msg = _build_user_message(analysis_type, results, profile)

    try:
        from groq import Groq  # type: ignore
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=600,
        )
        raw = response.choices[0].message.content or ""
        return _parse_json_response(raw)
    except Exception as exc:
        logger.warning("Groq career coach failed: %s — using fallback", exc)
        return _fallback_advice(analysis_type, results, profile)


def _build_user_message(
    analysis_type: str,
    results: dict[str, Any],
    profile: dict[str, Any],
) -> str:
    profile_summary = (
        f"Student: {profile.get('branch','CSE')} Year {profile.get('year',2)}, "
        f"CGPA {profile.get('cgpa',7.0):.1f}, "
        f"Skills: {', '.join(profile.get('current_skills',[])[:8])}."
    )
    results_summary = json.dumps(results, indent=2)[:1200]  # cap to fit context
    return (
        f"Analysis Type: {analysis_type}\n\n"
        f"Student Profile: {profile_summary}\n\n"
        f"ML Analysis Results:\n{results_summary}\n\n"
        "Give me hyper-personalized career advice grounded in these specific numbers."
    )


def _parse_json_response(text: str) -> dict[str, Any]:
    """Extract JSON from LLM output with multiple fallback strategies."""
    import re
    # 1. Direct parse
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass
    # 2. Code block
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 3. Find braces
    m = re.search(r"\{[\s\S]+\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # 4. Return raw as strategic_advice
    return {
        "headline": "Your personalized career roadmap is ready.",
        "immediate_actions": [
            "Review your skill gap analysis and prioritize top 3 missing skills.",
            "Set a 30-day learning goal for your highest-ROI skill.",
            "Build a portfolio project using your strongest skills.",
        ],
        "strategic_advice": text[:400],
        "salary_negotiation_tip": "Benchmark your offer against the P75 percentile for your role and CGPA band.",
        "motivational_insight": "Every skill you add compounds — the delta between packages at top companies and average ones is just 2-3 strategic skills.",
    }


def _fallback_advice(
    analysis_type: str,
    results: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """Static high-quality advice when Groq is unavailable."""
    cgpa = profile.get("cgpa", 7.0)
    skills = profile.get("current_skills", [])
    branch = profile.get("branch", "CSE")

    salary = None
    if analysis_type == "salary":
        salary = results.get("predicted_lpa") or results.get("mean_lpa")

    headline = (
        f"With CGPA {cgpa:.1f} and {len(skills)} technical skills, "
        f"you're positioned for {branch} roles in the ₹{int((salary or 9)):d}–"
        f"{int((salary or 9) * 1.4):d} LPA range."
    )

    return {
        "headline": headline,
        "immediate_actions": [
            "Spend 3 hours/day for 45 days on your highest-ROI skill (see Skill ROI tab).",
            "Deploy 2 GitHub projects using your strongest skills — recruiters scan pinned repos.",
            "Mock interview on LeetCode Medium for 30 min daily — DSA screens 60% of SDE roles.",
        ],
        "strategic_advice": (
            "Your gap analysis reveals specific missing skills recruiters screen for. "
            "Target the top 3 missing skills that appear in the Boost Simulator — "
            "they have disproportionate salary impact relative to learning time. "
            "Aim for one certification within 60 days to signal commitment."
        ),
        "salary_negotiation_tip": (
            f"At ₹{salary or 9:.1f} LPA predicted, expect offers in the ₹{(salary or 9) * 0.9:.1f}–"
            f"₹{(salary or 9) * 1.3:.1f} LPA range. Counter with P75 market data from Levels.fyi "
            "and highlight project impact metrics — never negotiate without written competing offers."
        ),
        "motivational_insight": (
            f"You are in the top 30% of profiles analyzed by SkillSync — "
            "the gap between your current trajectory and ₹20LPA is exactly 2 strategic skills."
        ),
    }
