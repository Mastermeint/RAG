"""Load processed chunks and rank them with BM25."""

import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rag_mvp.credits import is_credit_line

CREDIT_MAX_WORDS = 15
_STOPWORDS = frozenset(
    "a an and are as at be been by can could did do does for from had has have how i in into is it its of on "
    "or that the their there these this those to was were what when where which who whom why will with would "
    "you your between about than then them they not no".split())


@dataclass(frozen=True)
class Passage:
    source: str
    index: int
    text: str
    page: int | None = None
    page_end: int | None = None
    page_label: int | None = None
    type: str = "text"
    section_path: tuple[str, ...] = ()
    images: tuple[str, ...] = ()
    id: str = ""
    chapter: str = ""
    page_label_end: int | None = None
    low_quality: bool = False
    table_degraded: bool = False
    credit_only: bool = False

    @property
    def search_text(self) -> str:
        return " ".join(self.section_path) + " " + self.text


def stem(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith(("sses", "shes", "ches", "xes")):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return [stem(t) for t in re.findall(r"[a-z0-9]+", text) if t not in _STOPWORDS]


def _windows(text: str, size: int, overlap: int) -> list[str]:
    pieces, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            end = text.rfind(" ", start + size // 2, end) + 1 or end  # break at whitespace, not mid-word
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
        while start < len(text) and text[start - 1] != " ":  # restart on a word boundary
            start += 1
    return [p for p in pieces if p]


def load_passages(data_dir: Path, chunk_size: int = 1000, overlap: int = 150) -> list[Passage]:
    passages: list[Passage] = []
    for path in sorted(data_dir.rglob("*")):
        suffix = path.suffix.lower()
        if not path.is_file() or suffix not in {".txt", ".md", ".jsonl"}:
            continue
        if suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                kind, text = r.get("type", "text"), r["text"]
                passages.append(Passage(
                    r["source"], len(passages) + 1, text, r.get("page"), r.get("page_end"),
                    r.get("page_label"), kind, tuple(r.get("section_path", ())),
                    tuple(r.get("images", ())), r.get("id", ""), r.get("chapter", ""), r.get("page_label_end"),
                    r.get("low_quality", False), r.get("table_structure_degraded", False),
                    kind == "figure" and len(text.split()) < CREDIT_MAX_WORDS and is_credit_line(text)))
        else:
            name = path.relative_to(data_dir).as_posix()
            for number, piece in enumerate(_windows(path.read_text(encoding="utf-8").strip(), chunk_size, overlap), 1):
                passages.append(Passage(name, number, piece))
    return passages


def load_corpus(data_dir: Path | None = None, base: Path = Path(".")) -> list[Passage]:
    """The passages every entry point searches: `data_dir` if given, else raw text plus processed books."""
    if data_dir:
        return load_passages(data_dir)
    return load_passages(base / "data" / "raw") + load_passages(base / "data" / "processed" / "books")


@dataclass(frozen=True)
class RankingConfig:
    """Multipliers applied to BM25 scores. 1.0 is neutral; 0 hides a passage. Use dataclasses.replace to vary."""

    low_quality_weight: float = 1.0
    table_degraded_weight: float = 1.0
    credit_only_weight: float = 1.0
    type_weights: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        weights = {"low_quality_weight": self.low_quality_weight, "table_degraded_weight": self.table_degraded_weight,
                   "credit_only_weight": self.credit_only_weight,
                   **{f"type_weights.{k}": v for k, v in self.type_weights.items()}}
        for name, value in weights.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value >= 0:
                raise ValueError(f"{name} must be a number >= 0, got {value!r}")

    def multiplier(self, passage: "Passage") -> float:
        weight = self.type_weights.get(passage.type, 1.0)
        if passage.low_quality:
            weight *= self.low_quality_weight
        if passage.table_degraded:
            weight *= self.table_degraded_weight
        if passage.credit_only:
            weight *= self.credit_only_weight
        return weight


NEUTRAL = RankingConfig(low_quality_weight=1.0, table_degraded_weight=1.0, credit_only_weight=1.0, type_weights={})


class Index:
    """In-memory BM25 over passage text plus its section titles."""

    def __init__(self, passages: list[Passage], k1: float = 1.5, b: float = 0.75,
                 ranking: RankingConfig | None = None):
        self.passages, self.k1, self.b = passages, k1, b
        self.ranking = ranking or RankingConfig()
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.lengths = []
        for position, passage in enumerate(passages):
            counts = Counter(tokenize(passage.search_text))
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings.setdefault(term, []).append((position, count))
        self.average = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

    def search_scored(self, question: str, top_k: int = 3, types: set[str] | None = None,
                      ranking: RankingConfig | None = None) -> list[tuple[Passage, float]]:
        """Passages with final scores: BM25 times the ranking weights. `ranking` overrides the index default."""
        ranking = ranking or self.ranking
        n = len(self.passages)
        scores: dict[int, float] = {}
        for term in set(tokenize(question)):
            postings = self.postings.get(term, ())
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for position, count in postings:
                norm = count + self.k1 * (1 - self.b + self.b * self.lengths[position] / (self.average or 1))
                scores[position] = scores.get(position, 0.0) + idf * count * (self.k1 + 1) / norm
        for position in scores:
            scores[position] *= ranking.multiplier(self.passages[position])
        ranked = sorted((p for p in scores if scores[p] > 0), key=lambda p: (-scores[p], p))
        hits = [(self.passages[p], scores[p]) for p in ranked if types is None or self.passages[p].type in types]
        return hits[:top_k]

    def search(self, question: str, top_k: int = 3, types: set[str] | None = None,
               ranking: RankingConfig | None = None) -> list[Passage]:
        return [passage for passage, _ in self.search_scored(question, top_k, types, ranking)]


def retrieve(question: str, passages: "list[Passage] | Index", top_k: int = 3,
             types: set[str] | None = None) -> list[Passage]:
    index = passages if isinstance(passages, Index) else Index(passages)
    return index.search(question, top_k, types)
