"""Recognise photo-credit lines so ranking can weight them without deleting them."""

import re

_TITLE_BY = r"^\W{0,2}[\u201c\"].{1,150}?[\u201d\"]\s*(?:by|via|from)\b"
_MARKERS = (r"\bwikimedia\b", r"\bcreative commons\b", r"\bcc (?:by|sa|0)\b", r"\bcourtesy\b", r"\bphoto by\b",
            r"\bphoto credit\b", r"\bphotograph by\b", r"\bcredit:", r"\bsource:", r"\bpublic domain\b",
            r"\bvia (?:flickr|vimeo|youtube)\b", "\u00a9")
_CREDIT = re.compile("|".join((_TITLE_BY, *_MARKERS)), re.I)


def is_credit_line(text: str) -> bool:
    return bool(_CREDIT.search(text))
