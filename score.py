
from rapidfuzz import fuzz

def name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0

def compute_confidence(similarity: float, had_p452: bool, had_official_site: bool) -> float:
    base = similarity * 0.6 + (0.2 if had_p452 else 0.0) + (0.2 if had_official_site else 0.0)
    return min(1.0, round(base, 3))
