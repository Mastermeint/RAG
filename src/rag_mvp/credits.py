"""Recognise photo-credit lines so ranking can weight them without deleting them."""

import re

_TITLE_BY = r"^\W{0,2}[\u201c\"].{1,150}?[\u201d\"]\s*(?:by|via|from)\b"
_MARKERS = (r"\bwikimedia\b", r"\bcreative commons\b", r"\bcc (?:by|sa|0)\b", r"\bcourtesy\b", r"\bphoto by\b",
            r"\bphoto credit\b", r"\bphotograph by\b", r"\bcredit:", r"\bsource:", r"\bpublic domain\b",
            r"\bvia (?:flickr|vimeo|youtube)\b", "\u00a9")
_CREDIT = re.compile("|".join((_TITLE_BY, *_MARKERS)), re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def is_credit_line(text: str) -> bool:
    return bool(_CREDIT.search(text))


def is_credit_only(text: str) -> bool:
    """A credit with nothing else to say: every sentence of four or more words carries a credit marker.

    'A reef shark photographed at Roatan. CC BY 2.5' is a caption that ends in a credit, not a credit line."""
    plain = [s for s in _SENTENCE.split(text) if len(s.split()) >= 4 and not _CREDIT.search(s)]
    return is_credit_line(text) and not plain
