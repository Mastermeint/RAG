"""Group extracted page elements into section-aware, citable chunks."""

import re
from collections import Counter
from dataclasses import dataclass, field

from rag_mvp.textclean import join_lines, repair_ocr, strip_soft_hyphens, text_quality

_TERMINAL = re.compile(r"[.!?:;\"”’)\]]$")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z“\"(\[])")


@dataclass
class Element:
    kind: str  # paragraph | heading | table | figure | aside
    page: int
    page_end: int
    lines: list[str] = field(default_factory=list)
    size: float = 0.0
    caption: str = ""
    markdown: str = ""
    images: list[str] = field(default_factory=list)
    toc_path: list[str] = field(default_factory=list)
    chapter: str = ""
    page_label: int | None = None
    page_label_end: int | None = None
    poor: bool = False
    scanned: bool = False
    degraded: bool = False


def _words(text: str) -> int:
    return len(text.split())


def _render(element: Element, vocab) -> str:
    text = join_lines(element.lines, vocab)
    return repair_ocr(text, vocab) if element.scanned else text


def _merge_continuations(elements: list[Element]) -> list[Element]:
    """Rejoin a paragraph split by a column or page break, even when a figure or table floats in between."""
    merged: list[Element] = []
    open_para: Element | None = None
    for element in elements:
        if element.kind == "paragraph":
            first = element.lines[0].lstrip() if element.lines else ""
            if (open_para is not None and first[:1].islower() and 0 <= element.page - open_para.page_end <= 1
                    and open_para.chapter == element.chapter):
                open_para.lines.extend(element.lines)
                open_para.page_end = element.page_end
                open_para.page_label_end = element.page_label
                open_para.poor = open_para.poor or element.poor
                if _TERMINAL.search(open_para.lines[-1].rstrip()):
                    open_para = None
                continue
            merged.append(element)
            open_para = None if _TERMINAL.search(element.lines[-1].rstrip()) else element
        else:
            if element.kind == "heading":
                open_para = None
            merged.append(element)
    return merged


def _sentences(text: str, limit: int) -> list[str]:
    units: list[str] = []
    for sentence in _SENTENCE.split(text):
        words = sentence.split()
        for start in range(0, len(words), limit):
            units.append(" ".join(words[start:start + limit]))
    return units


def chunk_elements(elements: list[Element], vocab, doc_id: str, source: str, target_words: int = 180,
                   max_words: int = 380, overlap_words: int = 40) -> list[dict]:
    """Chunks never mix tables/figures with prose and never cross a heading or chapter boundary."""
    chunks: list[dict] = []
    heading = ""
    previous_scope: tuple | None = None

    state = {"units": [], "words": 0, "fresh": 0, "first": None, "last": None, "path": []}

    def emit(kind: str, text: str, first: Element, last: Element, path: list[str], images=None,
             degraded: bool = False) -> None:
        text = strip_soft_hyphens(text).strip()
        if not text or (kind == "text" and _words(text) < 8):
            return
        chunks.append({
            "id": f"{doc_id}-{len(chunks) + 1:05d}", "doc_id": doc_id, "source": source, "type": kind,
            "text": text, "section_path": path, "chapter": first.chapter,
            "page": first.page, "page_end": last.page_end,
            "page_label": first.page_label,
            "page_label_end": last.page_label_end if last.page_label_end is not None else last.page_label,
            "images": images or [], "low_quality": kind != "table" and text_quality(text)["poor"],
            "table_structure_degraded": degraded})

    def flush(carry: bool) -> None:
        if state["fresh"] and state["first"] is not None:
            emit("text", "\n\n".join(state["units"]), state["first"], state["last"], state["path"])
        tail: list[str] = []
        if carry and state["units"]:
            for sentence in reversed(_SENTENCE.split(state["units"][-1])):
                if sum(map(_words, tail)) + _words(sentence) > overlap_words:
                    break
                tail.insert(0, sentence)
        state.update(units=[" ".join(tail)] if tail else [], words=sum(map(_words, tail)), fresh=0,
                     first=state["last"] if tail else None)

    asides: list[Element] = []

    def emit_asides() -> None:
        """Margin notes on one page become their own chunk(s), so they neither split nor pad the prose around them."""
        group: list[Element] = []
        for element in asides + [None]:
            words = sum(_words(_render(e, vocab)) for e in group)
            if group and (element is None or words + _words(_render(element, vocab)) > max_words):
                emit("text", "\n\n".join(_render(e, vocab) for e in group), group[0], group[-1],
                     list(group[0].toc_path))
                group = []
            if element is not None:
                group.append(element)
        asides.clear()

    for element in _merge_continuations(elements):
        if element.kind == "aside":
            if asides and asides[-1].page != element.page:
                emit_asides()
            asides.append(element)
            continue
        emit_asides()
        scope = (tuple(element.toc_path[:2]), element.chapter)
        if scope != previous_scope:
            flush(carry=False)
            heading = ""
            previous_scope = scope
        if element.kind == "heading":
            flush(carry=False)
            heading = _render(element, vocab)
            continue
        path = list(element.toc_path)
        if heading and not any(p.lower() in heading.lower() or heading.lower() in p.lower() for p in path):
            path.append(heading)
        if element.kind == "figure":
            flush(carry=False)
            emit("figure", _render(element, vocab), element, element, path, element.images)
            continue
        if element.kind == "table":
            flush(carry=False)
            header, *rows = element.markdown.splitlines()
            head_rows = header + "\n" + rows[0] if rows else header
            body = rows[1:]
            group: list[str] = []
            for row in body + [None]:
                if row is None or (group and _words("\n".join(group + [row])) > max_words):
                    text = "\n".join([head_rows, *group])
                    emit("table", (element.caption + "\n\n" if element.caption else "") + text, element, element, path,
                         degraded=element.degraded)
                    group = []
                if row is not None:
                    group.append(row)
            continue
        text = _render(element, vocab)
        units = _sentences(text, max_words) if _words(text) > target_words else [text]
        for unit in units:
            unit_words = _words(unit)
            if state["fresh"] and state["words"] + unit_words > max_words:
                flush(carry=True)
            if state["first"] is None:
                state["first"] = element
            state["units"].append(unit)
            state["words"] += unit_words
            state["fresh"] += unit_words
            state["last"] = element
            state["path"] = path
            if state["words"] >= target_words:
                flush(carry=True)
    emit_asides()
    flush(carry=False)
    return chunks
