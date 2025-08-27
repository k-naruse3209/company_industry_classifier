
import re
import unicodedata

# Common corporate suffix patterns (JP/EN) to drop for normalization
_SUFFIX_PATTERNS = [
    r"\b株式会社\b",
    r"\b（株）\b",
    r"\b有限会社\b",
    r"\b（有）\b",
    r"\b合同会社\b",
    r"\b合資会社\b",
    r"\b合名会社\b",
    r"\b株式會社\b",
    r"\bInc\.?\b",
    r"\bIncorporated\b",
    r"\bCorp\.?\b",
    r"\bCorporation\b",
    r"\bCo\.?,?\s*Ltd\.?\b",
    r"\bCo\.?\b",
    r"\bLtd\.?\b",
    r"\bLLC\b",
    r"\bPLC\b",
    r"\bGmbH\b",
    r"\bS\.?A\.?\b",
    r"\bS\.?p\.?A\.?\b",
    r"\bAG\b",
]

_SUFFIX_RE = re.compile("|".join(_SUFFIX_PATTERNS), flags=re.IGNORECASE)

def zenkaku_to_hankaku(s: str) -> str:
    # Convert full-width to half-width for ASCII range
    return unicodedata.normalize("NFKC", s)

def normalize_company_name(name: str) -> str:
    if not name:
        return ""
    s = zenkaku_to_hankaku(name).strip()
    # Remove brackets and contents like (日本) or 【】
    s = re.sub(r"[\(\（][^\)\）]*[\)\）]", "", s)
    s = re.sub(r"[【［\[][^】\］\]]*[】\］\]]", "", s)
    # Remove common suffixes
    s = _SUFFIX_RE.sub("", s)
    # Remove leading/trailing punctuation and condense whitespace
    s = re.sub(r"[\"'’·・,，。｡、]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s
