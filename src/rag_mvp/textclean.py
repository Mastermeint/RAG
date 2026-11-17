"""Character-level text repair and page-quality scoring for text extracted from PDFs."""

import re
import unicodedata
from collections import Counter

SOFT_HYPHEN = "\xad"

_LIGATURES = {"\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl", "\ufb03": "ffi",
              "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st"}
_HYPHENS = "\u2010\u2011\u2012\u2212"  # hyphen, non-breaking hyphen, figure dash, minus sign
_SPACES = "\xa0\u2002\u2003\u2009\u200a\u202f\u2007\u2008\u2004\u2005\u2006"
_INVISIBLE = "\u200b\u200c\u200d\u2060\ufeff\x00"
_TRANSLATION = str.maketrans({
    **{k: v for k, v in _LIGATURES.items()},
    **{c: "-" for c in _HYPHENS},
    **{c: " " for c in _SPACES},
    **{c: None for c in _INVISIBLE},
})
_WORD = re.compile(r"[a-z]+(?:-[a-z]+)*")


def clean_chars(text: str) -> str:
    """Fix ligatures, odd hyphens/spaces and invisible characters; keeps soft hyphens for line joining."""
    text = text.translate(_TRANSLATION)
    return unicodedata.normalize("NFC", text)


def strip_soft_hyphens(text: str) -> str:
    return text.replace(SOFT_HYPHEN, "")


def collapse_space(text: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", " ", text).strip()


def build_vocabulary(lines: list[str]) -> Counter:
    """Lower-cased word counts from mid-line text, used to decide how to rejoin hyphenated line breaks."""
    vocab: Counter = Counter()
    for line in lines:
        vocab.update(_WORD.findall(strip_soft_hyphens(line).lower()))
    return vocab


def join_fragments(first: str, second: str, vocab: Counter | None = None) -> str:
    """Join two consecutive text lines, repairing a hyphenated word break between them."""
    first = first.rstrip()
    second = second.lstrip()
    if not first:
        return second
    if not second:
        return first
    if first.endswith(SOFT_HYPHEN) and not first.endswith("-" + SOFT_HYPHEN):
        return first[:-1] + second  # discretionary hyphen: the word was split here
    if first.endswith("-" + SOFT_HYPHEN):
        return first[:-1] + second  # explicit hyphen followed by a break opportunity: real compound
    if first.endswith("-") and len(first) > 1 and first[-2].isalpha() and second[:1].islower():
        head = re.search(r"([A-Za-z]+)-$", first)
        tail = re.match(r"[A-Za-z]+", second)
        if head and tail and vocab is not None:
            joined = (head.group(1) + tail.group(0)).lower()
            hyphenated = f"{head.group(1)}-{tail.group(0)}".lower()
            if vocab.get(joined, 0) == 0 and vocab.get(hyphenated, 0) > 0:
                return first + second  # a genuine compound such as "well-known" broken at its hyphen
        return first[:-1] + second
    if first.endswith("-") and len(first) > 1 and first[-2].isalpha():
        return first + second  # hyphen before a capital or digit: a compound such as Indo-Pacific
    return first + " " + second


def join_lines(lines: list[str], vocab: Counter | None = None) -> str:
    text = ""
    for line in lines:
        text = join_fragments(text, line, vocab) if text else line.strip()
    return collapse_space(strip_soft_hyphens(text))


_SUSPICIOUS = re.compile(r"[A-Za-z]+\d+[A-Za-z]+|[~^{}\\_<>]|\b[A-Za-z]\d\b")


def text_quality(text: str) -> dict:
    """Cheap OCR/encoding damage indicators; high `suspicious` or low `alpha` means unreliable text."""
    tokens = text.split()
    if not tokens:
        return {"chars": 0, "alpha": 0.0, "suspicious": 0.0, "poor": True}
    letters = sum(c.isalpha() for c in text)
    non_space = sum(not c.isspace() for c in text) or 1
    suspicious = sum(bool(_SUSPICIOUS.search(token)) for token in tokens) / len(tokens)
    alpha = letters / non_space
    return {"chars": len(text), "alpha": round(alpha, 3), "suspicious": round(suspicious, 3),
            "poor": len(text) >= 40 and (suspicious > 0.06 or alpha < 0.6)}


def damage(text: str) -> float:
    """Single score for comparing two readings of the same page: lower is cleaner."""
    q = text_quality(text)
    return q["suspicious"] + max(0.0, 0.6 - q["alpha"])


def is_noise(text: str) -> bool:
    """True for fragments such as scanned-figure speckle that carry no readable words."""
    letters = sum(c.isalpha() for c in text)
    return letters < 4 or (len(text) < 80 and letters / max(len(text), 1) < 0.5)


# (OCR misreading, intended letters): h read as I1/l1/11/ll, n as 11, m as rn, i/l as 1, o as 0.
_CONFUSIONS = (("i1", "h"), ("l1", "h"), ("11", "h"), ("ll", "h"), ("11", "n"), ("rn", "m"),
               ("1", "l"), ("1", "i"), ("0", "o"))
_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _repair_token(token: str, vocab: Counter) -> str:
    low = token.lower()
    seen = vocab.get(low, 0)
    if seen >= 3 or len(low) < 3:
        return token
    best, best_count = None, 0
    for wrong, right in _CONFUSIONS:
        start = low.find(wrong)
        while start != -1:
            candidate = low[:start] + right + low[start + len(wrong):]
            count = vocab.get(candidate, 0)
            # Only swap an unseen/rare token for a word the book itself uses very often.
            if count >= 20 and count >= 10 * max(seen, 1) and count > best_count:
                best, best_count = candidate, count
            start = low.find(wrong, start + 1)
    if best is None:
        return token
    return best.upper() if token.isupper() and len(token) > 1 else best.capitalize() if token[0].isupper() else best


def repair_ocr(text: str, vocab: Counter) -> str:
    """Fix systematic OCR letter confusions (TI1e -> The, rnay -> may) using the book's own vocabulary."""
    return _TOKEN.sub(lambda m: _repair_token(m.group(0), vocab), text)
