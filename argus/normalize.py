"""Value normalizers.

Everything the resolver compares passes through here first. Normalization is
lossy on purpose: case, punctuation, honorifics and formatting carry no
identity signal, so they get stripped before anything is compared.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime

_HONORIFICS = {
    "mr", "mrs", "ms", "miss", "dr", "prof", "sir", "madam", "mx",
    "rev", "hon", "capt", "col", "gen", "lt", "sgt",
}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "phd", "md", "esq"}

# Legal-form suffixes only. "Holdings", "Group", "Partners" and friends look
# like boilerplate but are part of the distinctive name: stripping them makes
# "Meridian Holdings" and "Meridian Group" identical, which is how
# over-normalization manufactures false merges.
_ORG_STOPWORDS = {
    "inc", "incorporated", "llc", "llp", "ltd", "limited", "co", "corp",
    "corporation", "company", "gmbh", "sa", "sas", "bv", "nv", "plc", "ag",
    "the", "and", "of",
}

# Nickname -> canonical given name. Deliberately small; a hand-curated table
# beats fuzzy matching for the handful of names that are genuinely irregular.
NICKNAMES = {
    "bill": "william", "billy": "william", "will": "william", "liam": "william",
    "bob": "robert", "bobby": "robert", "rob": "robert", "robbie": "robert",
    "dick": "richard", "rick": "richard", "ricky": "richard", "rich": "richard",
    "jim": "james", "jimmy": "james", "jamie": "james",
    "joe": "joseph", "joey": "joseph",
    "mike": "michael", "mickey": "michael", "mick": "michael",
    "tom": "thomas", "tommy": "thomas",
    "dave": "david", "davy": "david",
    "steve": "stephen", "steven": "stephen",
    "chris": "christopher", "kit": "christopher",
    "tony": "anthony",
    "ed": "edward", "eddie": "edward", "ted": "edward", "ned": "edward",
    "sam": "samuel", "sammy": "samuel",
    "dan": "daniel", "danny": "daniel",
    "matt": "matthew",
    "nick": "nicholas",
    "alex": "alexander", "sasha": "alexander",
    "kate": "katherine", "katie": "katherine", "kathy": "katherine",
    "cathy": "catherine", "cate": "catherine",
    "liz": "elizabeth", "beth": "elizabeth", "betty": "elizabeth",
    "eliza": "elizabeth", "lizzie": "elizabeth",
    "meg": "margaret", "maggie": "margaret", "peggy": "margaret",
    "sue": "susan", "susie": "susan",
    "jen": "jennifer", "jenny": "jennifer",
    "becky": "rebecca", "bex": "rebecca",
    "abby": "abigail",
    "nat": "natalie", "natasha": "natalia",
    "vicky": "victoria", "tori": "victoria",
    "andy": "andrew", "drew": "andrew",
    "pete": "peter",
    "ben": "benjamin", "benny": "benjamin",
    "greg": "gregory",
    "jeff": "jeffrey",
    "ken": "kenneth",
    "larry": "lawrence",
    "ray": "raymond",
    "ron": "ronald",
    "walt": "walter",
    "gabe": "gabriel",
    "isa": "isabella", "bella": "isabella",
    "sofia": "sophia",
    "yuri": "yuriy",
}


def strip_accents(value: str) -> str:
    """Fold accented characters to their ASCII base (José -> Jose)."""
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def norm_text(value) -> str:
    """Baseline text normalization: casefold, de-accent, collapse whitespace."""
    if value is None:
        return ""
    text = strip_accents(str(value)).casefold().strip()
    return re.sub(r"\s+", " ", text)


def norm_name(value) -> str:
    """Normalize a personal name.

    Drops honorifics and generational suffixes and expands nicknames. Token
    order is preserved: "Smith John" vs "John Smith" is the comparator's
    problem, not this function's.
    """
    text = norm_text(value)
    text = re.sub(r"[^a-z\s'-]", " ", text)
    tokens = [t.strip("'-") for t in text.split() if t.strip("'-")]
    tokens = [t for t in tokens if t not in _HONORIFICS and t not in _SUFFIXES]
    tokens = [NICKNAMES.get(t, t) for t in tokens]
    return " ".join(tokens)


def norm_org(value) -> str:
    """Normalize an organization name by dropping legal-form stopwords."""
    text = norm_text(value)
    # Drop periods before splitting so "S.A." collapses to the legal-form token
    # "sa" and is stripped, rather than surviving as two junk tokens "s" "a".
    text = text.replace(".", "")
    text = re.sub(r"[^a-z0-9\s&]", " ", text)
    tokens = [t for t in text.split() if t and t not in _ORG_STOPWORDS]
    return " ".join(tokens) or text.strip()


def norm_email(value) -> str:
    """Normalize an email address, including Gmail-style dot/plus folding."""
    text = norm_text(value)
    if "@" not in text:
        return text
    local, _, domain = text.rpartition("@")
    local = local.split("+", 1)[0]
    if domain in {"gmail.com", "googlemail.com"}:
        local = local.replace(".", "")
        domain = "gmail.com"
    return f"{local}@{domain}" if local else ""


def norm_phone(value) -> str:
    """Reduce a phone number to its last 10 significant digits."""
    digits = re.sub(r"\D", "", str(value or ""))
    if not digits:
        return ""
    if len(digits) > 10:
        digits = digits[-10:]
    return digits


def norm_id(value) -> str:
    """Normalize a government/tax/registration identifier."""
    return re.sub(r"[^a-z0-9]", "", norm_text(value))


_DATE_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d",
    "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y", "%Y%m%d",
)


def norm_date(value):
    """Parse a date written in any of several common formats.

    Returns a date or None. Ambiguous day/month orderings resolve ISO first,
    then day-first, matching the source conventions.
    """
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = re.sub(r"[,]", "", str(value).strip())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def norm_address(value) -> str:
    """Normalize a street address by expanding common abbreviations."""
    text = norm_text(value)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    expansions = {
        "st": "street", "rd": "road", "ave": "avenue", "av": "avenue",
        "blvd": "boulevard", "ln": "lane", "dr": "drive", "ct": "court",
        "apt": "apartment", "ste": "suite", "fl": "floor", "sq": "square",
        "pl": "place", "hwy": "highway", "pkwy": "parkway",
        "n": "north", "s": "south", "e": "east", "w": "west",
    }
    tokens = [expansions.get(t, t) for t in text.split()]
    return " ".join(tokens)


def soundex(token: str) -> str:
    """Classic Soundex code, used as a cheap phonetic blocking key."""
    token = re.sub(r"[^a-z]", "", norm_text(token))
    if not token:
        return ""
    codes = {
        **dict.fromkeys("bfpv", "1"),
        **dict.fromkeys("cgjkqsxz", "2"),
        **dict.fromkeys("dt", "3"),
        "l": "4",
        **dict.fromkeys("mn", "5"),
        "r": "6",
    }
    first = token[0]
    encoded = [codes.get(first, "")]
    for ch in token[1:]:
        code = codes.get(ch, "")
        # Vowels reset the previous code, so doubled sounds separated by a
        # vowel survive while genuine repeats collapse.
        if ch in "aeiouy":
            encoded.append("")
            continue
        if ch in "hw":
            continue
        if code and code != encoded[-1]:
            encoded.append(code)
    digits = "".join(c for c in encoded[1:] if c)
    return (first.upper() + digits + "000")[:4]


NORMALIZERS = {
    "text": norm_text,
    "name": norm_name,
    "org": norm_org,
    "email": norm_email,
    "phone": norm_phone,
    "id": norm_id,
    "date": norm_date,
    "address": norm_address,
    "number": lambda v: None if v in (None, "") else float(v),
    "raw": lambda v: v,
}


def normalize(kind: str, value):
    """Apply the normalizer registered under ``kind``."""
    return NORMALIZERS.get(kind, norm_text)(value)
