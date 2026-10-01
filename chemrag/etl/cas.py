"""CAS Registry Number normalisation and check-digit validation.

A CAS number is ``NNNNNNN-NN-R`` (2-7 digits, 2 digits, 1 check digit). The check digit R is
``sum(i * d_i) mod 10`` where d_i are the other digits read right-to-left and i starts at 1.
Validating it lets us repair messy strings ("13463677", "93 15 2", "CAS #79-81-2") without guessing.
"""

from __future__ import annotations

import re
from typing import Literal

CasStatus = Literal["valid", "repaired", "from_casid", "invalid", "missing", "sentinel_zero"]

CAS_RE = re.compile(r"^(\d{2,7})-(\d{2})-(\d)$")
_PREFIX_RE = re.compile(r"(?i)\b(?:cas\s*(?:no\.?|number|rn|#)?|rn)\s*[:#]?\s*")
_GROUPS_RE = re.compile(r"\d+")


def check_digit_ok(cas: str) -> bool:
    m = CAS_RE.match(cas)
    if not m:
        return False
    body = m.group(1) + m.group(2)
    total = sum(i * int(d) for i, d in enumerate(reversed(body), start=1))
    return total % 10 == int(m.group(3))


def format_digits(digits: str) -> str | None:
    """Re-hyphenate a bare digit string (5-10 digits) into CAS layout."""
    if not digits.isdigit() or not 5 <= len(digits) <= 10:
        return None
    return f"{digits[:-3]}-{digits[-3:-1]}-{digits[-1]}"


def normalize_cas(raw: str | None) -> tuple[str | None, CasStatus]:
    """Return ``(canonical_cas, status)`` for a raw CAS cell or user-typed CAS mention.

    - ``valid``: already canonical and check digit OK
    - ``repaired``: reformatted (whitespace, prefixes, separators, bare digits) and check digit OK
    - ``sentinel_zero``: the literal "0" used for trade-secret rows
    - ``missing``: empty
    - ``invalid``: could not be turned into a check-digit-valid CAS
    """
    if not isinstance(raw, str):  # None / NaN
        return None, "missing"
    s = str(raw).strip()
    if not s:
        return None, "missing"
    if s == "0":
        return None, "sentinel_zero"
    if CAS_RE.match(s) and check_digit_ok(s):
        return s, ("valid" if s == raw else "repaired")

    cleaned = _PREFIX_RE.sub("", s)
    groups = _GROUPS_RE.findall(cleaned)
    candidates: list[str] = []
    if len(groups) == 3:
        candidates.append("-".join(groups))
    # Bare digits are only re-hyphenated when the original had no separators at all
    # ("13463677", "79812"); "50-78-25" must not be silently reshaped into another CAS.
    if len(groups) == 1:
        f = format_digits(groups[0])
        if f:
            candidates.append(f)
    for c in candidates:
        if CAS_RE.match(c) and check_digit_ok(c):
            return c, "repaired"
    return None, "invalid"


def looks_like_cas(text: str) -> bool:
    return bool(re.fullmatch(r"\s*(?:cas\s*#?\s*)?\d{2,7}[-\s]?\d{2}[-\s]?\d\s*", text, flags=re.IGNORECASE))


def near_miss_suggestions(cas_like: str, known: list[str], limit: int = 3) -> list[str]:
    """Known CAS numbers within one digit edit of a mistyped CAS (for 'did you mean')."""
    digits = re.sub(r"\D", "", cas_like)
    out: list[str] = []
    for k in known:
        kd = k.replace("-", "")
        if len(kd) == len(digits) and sum(a != b for a, b in zip(kd, digits)) == 1:
            out.append(k)
        elif abs(len(kd) - len(digits)) == 1:
            short, long_ = sorted((kd, digits), key=len)
            if any(long_[:i] + long_[i + 1:] == short for i in range(len(long_))):
                out.append(k)
    return out[:limit]
