"""Text normalisation shared by the ETL, the resolver and the evaluation ground truth."""

from __future__ import annotations

import re
import unicodedata

NULL_PLACEHOLDERS = {"na", "n/a", "none", "null", "-", ""}

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s&]")
_LEGAL = re.compile(
    r"\b(inc|incorporated|llc|l\s?l\s?c|ltd|limited|corp|corporation|co|company|lp|l\s?p|plc|gmbh|sa|s\s?a|ag|bv|"
    r"pty|srl|spa|kk|entity)\b"
)


def clean_text(value: object) -> str | None:
    """Trim + collapse internal whitespace; map placeholders like 'NA'/'None' to None."""
    if not isinstance(value, str):
        return None
    s = _WS.sub(" ", str(value)).strip()
    if s.casefold() in NULL_PLACEHOLDERS:
        return None
    return s


def norm_key(value: str | None) -> str:
    """Case/accents/punctuation-insensitive key used for grouping and exact alias matching."""
    if not isinstance(value, str) or not value:
        return ""
    s = unicodedata.normalize("NFKD", value)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.casefold().replace("’", "'").replace("'", "")
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def company_match_key(value: str | None) -> str:
    """Looser key for *matching* company mentions: also drops legal suffixes (Inc, LLC, ...)."""
    s = norm_key(value).replace("&", " and ")
    s = _LEGAL.sub(" ", s)
    return _WS.sub(" ", s).strip()
