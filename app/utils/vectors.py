"""Utility helpers for working with pgvector columns and cosine similarity."""
from __future__ import annotations

from math import sqrt
from typing import List, Optional, Sequence


def parse_pgvector(value: Optional[str]) -> Optional[List[float]]:
    """Parse a textual pgvector representation ("[1,2,3]") into a list of floats."""
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if not parts:
        return []
    try:
        return [float(item) for item in parts]
    except ValueError:
        return None


def format_pgvector(values: Sequence[float]) -> str:
    """Format a sequence of floats into pgvector textual representation."""
    return "[" + ",".join(f"{float(v):.7f}" for v in values) + "]"


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine similarity in the range [0, 1]."""
    if not a or not b:
        return 0.0
    length = min(len(a), len(b))
    if length == 0:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for i in range(length):
        av = float(a[i])
        bv = float(b[i])
        dot += av * bv
        norm_a += av * av
        norm_b += bv * bv
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    denom = sqrt(norm_a) * sqrt(norm_b)
    if denom <= 0.0:
        return 0.0
    sim = dot / denom
    if sim < 0.0:
        return 0.0
    if sim > 1.0:
        return 1.0
    return sim
