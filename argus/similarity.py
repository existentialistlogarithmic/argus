"""String and value similarity functions.

These return a score in [0.0, 1.0]. They are the only place where "how alike
are these two values" is decided; the resolver turns those scores into
match/no-match evidence via the Fellegi-Sunter model in ``resolve.py``.
"""

from __future__ import annotations

from datetime import date


def jaro(a: str, b: str) -> float:
    """Jaro similarity: transposition-aware character agreement."""
    if a == b:
        return 1.0 if a else 0.0
    len_a, len_b = len(a), len(b)
    if not len_a or not len_b:
        return 0.0

    window = max(len_a, len_b) // 2 - 1
    if window < 0:
        window = 0

    a_matched = [False] * len_a
    b_matched = [False] * len_b
    matches = 0

    for i, ch in enumerate(a):
        start = max(0, i - window)
        end = min(i + window + 1, len_b)
        for j in range(start, end):
            if b_matched[j] or b[j] != ch:
                continue
            a_matched[i] = b_matched[j] = True
            matches += 1
            break

    if not matches:
        return 0.0

    # Count transpositions: matched characters that appear out of order.
    transpositions = 0
    k = 0
    for i in range(len_a):
        if not a_matched[i]:
            continue
        while not b_matched[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1
    transpositions //= 2

    return (
        matches / len_a + matches / len_b + (matches - transpositions) / matches
    ) / 3.0


def jaro_winkler(a: str, b: str, prefix_weight: float = 0.1) -> float:
    """Jaro similarity boosted for shared leading characters.

    Names agree on their prefix far more often than chance, so the boost is a
    genuine signal rather than a tuning knob.
    """
    base = jaro(a, b)
    if base < 0.7:
        return base
    prefix = 0
    for ch_a, ch_b in zip(a[:4], b[:4]):
        if ch_a != ch_b:
            break
        prefix += 1
    return base + prefix * prefix_weight * (1 - base)


def levenshtein(a: str, b: str) -> int:
    """Edit distance, computed with a rolling row to stay O(min(len)) in space."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        current = [i]
        for j, ch_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,          # deletion
                current[j - 1] + 1,       # insertion
                previous[j - 1] + (ch_a != ch_b),  # substitution
            ))
        previous = current
    return previous[-1]


def edit_ratio(a: str, b: str) -> float:
    """Normalized edit similarity."""
    if not a and not b:
        return 0.0
    longest = max(len(a), len(b))
    return 1.0 - levenshtein(a, b) / longest


def token_set_ratio(a: str, b: str) -> float:
    """Jaccard overlap of whitespace tokens; order-insensitive."""
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _token_score(a: str, b: str) -> float:
    """Score one token against another, treating initials specially."""
    if len(a) == 1 or len(b) == 1:
        # An initial agreeing with a full token is strong but not conclusive.
        return 0.9 if a[0] == b[0] else 0.0
    return jaro_winkler(a, b)


def _align_tokens(tokens_a: list[str], tokens_b: list[str]) -> tuple[list[float], int]:
    """Greedily pair tokens best-first, returning pair scores and the long length.

    Pairing must be best-first rather than left-to-right. Walking the tokens in
    order lets a middle initial consume the surname slot -- which is how
    "Robert J Whitlock" ends up not matching "Bob Whitlock" -- so every
    candidate pair is scored up front and the strongest available pair is taken
    each round.
    """
    if not tokens_a or not tokens_b:
        return [], max(len(tokens_a), len(tokens_b))

    scored = [
        (_token_score(a, b), i, j)
        for i, a in enumerate(tokens_a)
        for j, b in enumerate(tokens_b)
    ]
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_a: set[int] = set()
    used_b: set[int] = set()
    scores: list[float] = []
    for score, i, j in scored:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        scores.append(score)
    return scores, max(len(tokens_a), len(tokens_b))


def name_similarity(a: str, b: str) -> float:
    """Compare two normalized personal names.

    Order-insensitive, initial-aware, and forgiving of missing middle names --
    source systems drop them constantly -- but not of a differing given name.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    scores, longest = _align_tokens(a.split(), b.split())
    if not scores:
        return 0.0
    paired = sum(scores) / len(scores)
    # A token the other name simply lacks is weak evidence of difference; a
    # missing middle name is far more common than a different person.
    coverage = len(scores) / longest
    return max(paired * (1.0 - 0.15 * (1.0 - coverage)), token_set_ratio(a, b))


def org_similarity(a: str, b: str) -> float:
    """Compare two normalized organization names.

    Deliberately harsher than ``name_similarity``: an unpaired token scores
    zero rather than being discounted. Company names share leading words as a
    matter of course -- "Meridian Holdings" and "Meridian Trading" are
    unrelated firms -- so character-level similarity with a prefix bonus is
    exactly the wrong instrument. What matters is whether *every* token
    corresponds.
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    scores, longest = _align_tokens(a.split(), b.split())
    if not scores or not longest:
        return 0.0
    return sum(scores) / longest


def date_similarity(a, b, tolerance_days: int = 3) -> float:
    """Compare two dates, decaying with the gap between them.

    Transposed digits in a date of birth produce small gaps far more often than
    a genuinely different person does, so near-misses keep partial credit.
    """
    if not isinstance(a, date) or not isinstance(b, date):
        return 0.0
    if a == b:
        return 1.0
    gap = abs((a - b).days)
    # A day/month swap is the classic data-entry error; treat it as near-exact.
    if a.day == b.month and a.month == b.day and a.year == b.year:
        return 0.92
    if gap <= tolerance_days:
        return 1.0 - 0.1 * gap
    if gap <= 366:
        return max(0.0, 0.6 - gap / 800.0)
    return 0.0


def exact(a, b) -> float:
    """Strict equality on non-empty values."""
    if a in (None, "") or b in (None, ""):
        return 0.0
    return 1.0 if a == b else 0.0


def numeric_similarity(a, b, scale: float = 1.0) -> float:
    """Similarity of two numbers, decaying over ``scale`` units of difference."""
    if a is None or b is None:
        return 0.0
    gap = abs(float(a) - float(b))
    return max(0.0, 1.0 - gap / scale) if scale else float(gap == 0)


COMPARATORS = {
    "name": name_similarity,
    "org": org_similarity,
    "text": lambda a, b: jaro_winkler(a, b) if a and b else 0.0,
    "exact": exact,
    "email": exact,
    "phone": exact,
    "id": exact,
    "date": date_similarity,
    "address": lambda a, b: token_set_ratio(a, b) if a and b else 0.0,
    "edit": lambda a, b: edit_ratio(a, b) if a and b else 0.0,
    "number": numeric_similarity,
}


def compare(kind: str, a, b) -> float:
    """Apply the comparator registered under ``kind``."""
    return COMPARATORS.get(kind, COMPARATORS["text"])(a, b)
