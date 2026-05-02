"""
helpers.py
Shared utility functions used across the project.
"""

import re
from typing import List


def normalize_symptom(symptom: str) -> str:
    """Lowercase, strip whitespace, replace spaces with underscores."""
    return symptom.strip().lower().replace(" ", "_")


def normalize_symptoms(symptoms: List[str]) -> List[str]:
    return [normalize_symptom(s) for s in symptoms if s.strip()]


def parse_symptoms_from_input(raw_input: str) -> List[str]:
    """
    Accept a comma-separated or newline-separated string of symptoms
    and return a clean list.
    Example:
        "Fever, Headache, fatigue" → ["fever", "headache", "fatigue"]
    """
    parts = re.split(r"[,\n;]+", raw_input)
    return normalize_symptoms(parts)


def format_disease_list(diseases: List[dict]) -> str:
    """Pretty-format a list of disease dicts for display."""
    if not diseases:
        return "No matching diseases found."
    lines = []
    for i, d in enumerate(diseases, 1):
        matched = ", ".join(d.get("matched_symptoms", [])) or "—"
        lines.append(f"{i}. {d.get('name', d.get('disease', 'Unknown'))}")
        lines.append(f"   Matched symptoms: {matched}")
    return "\n".join(lines)


def truncate(text: str, max_chars: int = 300) -> str:
    """Truncate long strings for display purposes."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " …"
