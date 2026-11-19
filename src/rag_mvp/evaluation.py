"""Retrieval evaluation: labelled questions, metrics with bootstrap intervals, diagnostics and paired comparison.

Run with `uv run python -m rag_mvp.evaluation --help`."""

import argparse
import hashlib
import json
import random
import re
import statistics
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from rag_mvp.console import use_utf8
from rag_mvp.embeddings import (DEFAULT_MODE, DEFAULT_MODEL, LEXICAL_WEIGHT, MODELS, MODES, ModelMissing,
                                VectorsMissing, build_index, describe)
from rag_mvp.retrieval import NEUTRAL, Index, Passage, RankingConfig, load_corpus, tokenize

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_PATH = ROOT / "eval" / "questions.jsonl"

CATEGORIES = ("lexical", "paraphrase", "student", "table", "figure", "multi_book", "unanswerable")
MIN_COUNTS = {"lexical": 20, "paraphrase": 30, "student": 12, "table": 12, "figure": 12, "multi_book": 12,
              "unanswerable": 10}
SPLITS = ("dev", "holdout", "holdout2")  # holdout was used once (2026-11-17); holdout2 is the fresh final check
KS = (3, 5, 10)
DEPTH = 100
MAX_GOLD_PAGES, MAX_ANCHORS, MAX_ANCHOR_WORDS = 3, 3, 6
MAX_PARAPHRASE_OVERLAP = 0.34  # share of query terms present in the gold chunk
MIN_LEXICAL_OVERLAP = 0.5
VOCABULARY_OVERLAP = 0.34  # below this a miss is blamed on vocabulary, not ranking
MIN_NET_QUESTIONS = 2
DESCRIPTIVE_BELOW = 15
MIN_BOOK_QUESTIONS, MAX_FIRST_QUARTER_SHARE = 10, 0.4
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 2000, 0
SELFTEST_SAMPLE, SELFTEST_SEED, SELFTEST_TOP, SELFTEST_MIN_WORDS = 200, 7, 5, 40
SELFTEST_MIN_SHARE = 0.97  # measured 99.0% on the real corpus


@dataclass(frozen=True)
class Gold:
    source: str
    pages: tuple[int, int]
    anchor: tuple[str, ...]


@dataclass(frozen=True)
class Question:
    id: str
    split: str
    category: str
    question: str
    gold: tuple[Gold, ...]
    notes: str = ""


def normalize(text: str) -> str:
    """Lowercase, fold accents, collapse everything but ASCII letters and digits; padded so matches respect word edges."""
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).lower()
    return " " + re.sub(r"[^a-z0-9]+", " ", text).strip() + " "


def load_questions(path: Path) -> list[Question]:
    questions = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            gold = tuple(Gold(g["source"], tuple(g["pages"]), tuple(g.get("anchor", ()))) for g in r.get("gold", []))
            questions.append(Question(r["id"], r["split"], r["category"], r["question"], gold, r.get("notes", "")))
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"{path}:{number}: malformed question ({error!r})") from None
    return questions


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Matcher:
    """Decides whether a passage is a correct answer, caching each passage's normalised text."""

    def __init__(self, passages: list[Passage], mode: str = "anchor"):
        self.mode, self._norm = mode, {}
        self.by_source: dict[str, list[Passage]] = {}
        for passage in passages:
            self.by_source.setdefault(passage.source, []).append(passage)

    def text(self, passage: Passage) -> str:
        key = id(passage)
        if key not in self._norm:
            self._norm[key] = normalize(passage.text)
        return self._norm[key]

    def overlaps(self, passage: Passage, gold: Gold) -> bool:
        start = passage.page or 0
        return passage.source == gold.source and start <= gold.pages[1] and (passage.page_end or start) >= gold.pages[0]

    def anchored(self, passage: Passage, gold: Gold) -> bool:
        text = self.text(passage)
        return any(normalize(a) in text for a in gold.anchor if a.strip())

    def hit(self, passage: Passage, gold: Gold) -> bool:
        return self.overlaps(passage, gold) and (self.mode == "page" or self.anchored(passage, gold))

    def any_hit(self, passage: Passage, golds: tuple[Gold, ...]) -> bool:
        return any(self.hit(passage, g) for g in golds)

    def gold_passages(self, question: Question) -> list[Passage]:
        return [p for g in question.gold for p in self.by_source.get(g.source, ()) if self.hit(p, g)]


def term_share(question_text: str, passages: list[Passage]) -> float:
    """Share of the question's unique terms that occur anywhere in the given passages."""
    terms = set(tokenize(question_text))
    if not terms or not passages:
        return 0.0
    have = {t for p in passages for t in tokenize(p.search_text)}
    return len(terms & have) / len(terms)


# ---- running -----------------------------------------------------------------------------------------------------

_EXPOSURE = {
    "figure": lambda p: p.type == "figure",
    "table": lambda p: p.type == "table",
    "low_quality": lambda p: p.low_quality,
    "table_degraded": lambda p: p.table_degraded,
    "credit_only": lambda p: p.credit_only,
}


@dataclass
class Outcome:
    question: Question
    rank: int | None  # first matching result within DEPTH, else None
    top1: float
    top: list[tuple[Passage, float]]
    exposure: dict[str, float]


def run_questions(index: Index, questions: list[Question], matcher: Matcher, **search_options) -> list[Outcome]:
    outcomes = []
    for q in questions:
        hits = index.search_scored(q.question, DEPTH, **search_options)
        rank = next((i for i, (p, _) in enumerate(hits, 1) if matcher.any_hit(p, q.gold)), None) if q.gold else None
        top = hits[:10]
        exposure = {name: (sum(map(flag, (p for p, _ in top))) / len(top) if top else 0.0)
                    for name, flag in _EXPOSURE.items()}
        outcomes.append(Outcome(q, rank, hits[0][1] if hits else 0.0, hits[:3], exposure))
    return outcomes


# ---- metrics -----------------------------------------------------------------------------------------------------

def vector(rank: int | None) -> tuple[float, float, float, float]:
    r = rank if rank is not None else float("inf")
    return tuple(float(r <= k) for k in KS) + (1.0 / r if r <= 10 else 0.0,)


def metric_names() -> list[str]:
    return [f"recall@{k}" for k in KS] + ["mrr@10"]


def bootstrap(vectors: list[tuple], resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> list[tuple]:
    """95% percentile interval of each metric's mean over questions."""
    if not vectors:
        return [(0.0, 0.0)] * len(metric_names())
    rng, n = random.Random(seed), len(vectors)
    means = [[] for _ in metric_names()]
    for _ in range(resamples):
        sample = rng.choices(vectors, k=n)
        for m in range(len(means)):
            means[m].append(sum(v[m] for v in sample) / n)
    low, high = round(0.025 * (resamples - 1)), round(0.975 * (resamples - 1))
    return [(sorted(m)[low], sorted(m)[high]) for m in means]


def summarize(outcomes: list[Outcome]) -> dict:
    vectors = [vector(o.rank) for o in outcomes]
    n = len(vectors)
    intervals = bootstrap(vectors)
    out = {"n": n, "descriptive_only": n < DESCRIPTIVE_BELOW}
    for m, name in enumerate(metric_names()):
        out[name] = {"value": sum(v[m] for v in vectors) / n if n else 0.0, "ci": list(intervals[m])}
    return out


def group_metrics(outcomes: list[Outcome]) -> dict:
    answerable = [o for o in outcomes if o.question.gold]
    groups = {"overall": summarize(answerable)}
    for name, key in (("category", lambda o: [o.question.category]), ("split", lambda o: [o.question.split]),
                      ("source", lambda o: sorted({g.source for g in o.question.gold}))):
        keys = sorted({k for o in answerable for k in key(o)})
        groups[name] = {k: summarize([o for o in answerable if k in key(o)]) for k in keys}
    return groups


def percentile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = share * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def unanswerable_report(outcomes: list[Outcome]) -> dict:
    answer = [o.top1 for o in outcomes if o.question.gold]
    none = [o.top1 for o in outcomes if not o.question.gold]
    floor = percentile(answer, 0.10)
    return {"answerable_top1_median": statistics.median(answer) if answer else 0.0,
            "unanswerable_top1_median": statistics.median(none) if none else 0.0,
            "answerable_top1_p10": floor,
            "unanswerable_above_answerable_p10": (sum(s > floor for s in none) / len(none)) if none else 0.0,
            "n_unanswerable": len(none)}


def exposure_report(outcomes: list[Outcome]) -> dict:
    return {name: statistics.fmean(o.exposure[name] for o in outcomes) if outcomes else 0.0
            for name in _EXPOSURE}


def short_source(source: str) -> str:
    return source if len(source) <= 30 else source[:27] + "..."


def misses(outcomes: list[Outcome], matcher: Matcher) -> list[dict]:
    found = []
    for o in outcomes:
        if not o.question.gold or (o.rank is not None and o.rank <= 10):
            continue
        gold_chunks = matcher.gold_passages(o.question)
        if not gold_chunks:
            hint = "label"
        elif term_share(o.question.question, gold_chunks) < VOCABULARY_OVERLAP:
            hint = "vocabulary"
        else:
            hint = "ranking"
        found.append({
            "id": o.question.id, "category": o.question.category, "question": o.question.question,
            "rank": o.rank, "cause_hint": hint,
            "top3": [{"source": short_source(p.source), "pages": [p.page, p.page_end or p.page], "type": p.type,
                      "score": round(s, 3)} for p, s in o.top]})
    return found


# ---- fingerprint and report --------------------------------------------------------------------------------------

def corpus_fingerprint(root: Path = ROOT) -> list[dict]:
    books = []
    for path in sorted((root / "data" / "processed" / "books").glob("*.manifest.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        books.append({"source": m["source"], "sha256": m["sha256"], "extractor_version": m["extractor_version"],
                      "manifest_schema": m.get("manifest_schema", 1), "chunks": m["chunks"]})
    return books


def parse_ranking(items: list[str] | None) -> RankingConfig:
    """Apply `key=value` overrides (dotted keys for type_weights) on top of the defaults; 'neutral' resets."""
    config = RankingConfig()
    for item in items or []:
        if item == "neutral":
            config = NEUTRAL
            continue
        key, sep, raw = item.partition("=")
        config = apply_override(config, key.strip(), raw.strip()) if sep else _bad(item)
    return config


def _bad(item: str):
    raise ValueError(f"ranking override must look like key=value or 'neutral', got {item!r}")


def apply_override(config: RankingConfig, key: str, raw: str) -> RankingConfig:
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{key}: {raw!r} is not a number") from None
    if key.startswith("type_weights."):
        kind = key.split(".", 1)[1]
        if kind not in ("text", "table", "figure"):
            raise ValueError(f"unknown chunk type in {key!r}; use text, table or figure")
        return replace(config, type_weights={**config.type_weights, kind: value})
    names = {f.name for f in fields(RankingConfig)} - {"type_weights"}
    if key not in names:
        raise ValueError(f"unknown ranking key {key!r}; use {', '.join(sorted(names))} or type_weights.<type>")
    return replace(config, **{key: value})


def evaluate(index: Index, questions: list[Question], passages: list[Passage], questions_path: Path, split: str,
             match: str, root: Path = ROOT, ranking: RankingConfig | None = None) -> dict:
    chosen = [q for q in questions if q.split == split]
    matcher = Matcher(passages, match)
    outcomes = run_questions(index, chosen, matcher, ranking=ranking)
    return {
        "split": split, "match": match, "ranking": asdict(ranking or index.ranking), "retrieval": describe(index),
        "fingerprint": {"corpus": corpus_fingerprint(root), "questions_sha256": file_sha256(questions_path)},
        "metrics": group_metrics(outcomes), "exposure": exposure_report(outcomes),
        "unanswerable": unanswerable_report(outcomes), "misses": misses(outcomes, matcher),
        "questions": [{"id": o.question.id, "category": o.question.category, "rank": o.rank, "top1": round(o.top1, 4)}
                      for o in outcomes],
    }


def sweep(index: Index, questions: list[Question], passages: list[Passage], questions_path: Path, split: str,
          match: str, base: RankingConfig, key: str, values: list[str], reference: dict) -> list[str]:
    """One row per value of one ranking key, each compared with `reference` using the same verdict as --compare."""
    rows = [f"sweep {key} on {split} (reference: {reference['ranking']})",
            f"  {'value':<8}" + "".join(f"{n:<10}" for n in metric_names()) + "net@10  verdict"]
    for raw in values:
        result = evaluate(index, questions, passages, questions_path, split, match, ranking=apply_override(base, key, raw))
        overall, c = result["metrics"]["overall"], compare(reference, result, force=True)
        rows.append(f"  {raw:<8}" + "".join(f"{overall[n]['value']:<10.3f}" for n in metric_names())
                    + f"{c['net@10']:+d}      {c['verdict']}")
    rows.append(f"{len(values)} configurations tried on {split}; the best of {len(values)} is optimistic, "
                f"so only a verdict of 'improved' justifies a change")
    return rows


def fmt_metric(m: dict) -> str:
    low, high = m["ci"]
    return f"{m['value']:.2f} [{low:.2f}-{high:.2f}]"


def format_metrics(title: str, rows: dict) -> list[str]:
    lines = [title, f"  {'group':<34}{'n':>4}  " + "  ".join(f"{n:<17}" for n in metric_names())]
    for name, s in rows.items():
        mark = "  descriptive only" if s["descriptive_only"] else ""
        lines.append(f"  {short_source(name):<34}{s['n']:>4}  " + "  ".join(f"{fmt_metric(s[n]):<17}"
                                                                          for n in metric_names()) + mark)
    return lines


def format_report(result: dict, show_misses: bool = True) -> str:
    metrics = result["metrics"]
    retrieval = result.get("retrieval", {"mode": "bm25"})
    lines = [f"split={result['split']} match={result['match']} retrieval={json.dumps(retrieval)} "
             f"ranking={json.dumps(result['ranking'])}",
             *format_metrics("Overall (answerable questions)", {"overall": metrics["overall"]}),
             *format_metrics("By category", metrics["category"]), *format_metrics("By source", metrics["source"]),
             "Exposure in the top 10 (mean share): " + ", ".join(f"{k} {v:.1%}" for k, v in result["exposure"].items())]
    u = result["unanswerable"]
    above = u["unanswerable_above_answerable_p10"]
    verdict = ("so on this sample the top score separates 'no answer' from 'answer' (too few questions to rely on)"
               if above == 0 else "so the score alone cannot separate 'no answer' from 'answer'")
    lines.append(f"Top-1 score: answerable median {u['answerable_top1_median']:.4g}, unanswerable median "
                 f"{u['unanswerable_top1_median']:.4g}; {above:.0%} of "
                 f"{u['n_unanswerable']} unanswerable questions score above the answerable 10th percentile "
                 f"({u['answerable_top1_p10']:.4g}), {verdict}")
    if show_misses:
        lines.append(f"Misses at 10: {len(result['misses'])}")
        for m in result["misses"]:
            rank = m["rank"] if m["rank"] else f">{DEPTH}"
            lines.append(f"  {m['id']} [{m['category']}] hint={m['cause_hint']} rank={rank}: {m['question']}")
            for t in m["top3"]:
                lines.append(f"      {t['source']} p.{t['pages'][0]}-{t['pages'][1]} {t['type']} {t['score']:.4g}")
    return "\n".join(lines)


# ---- comparison --------------------------------------------------------------------------------------------------

def compare(a: dict, b: dict, force: bool = False) -> dict:
    """Paired comparison of two saved runs, with one verdict (see MIN_NET_QUESTIONS)."""
    if not force:
        for key, label in (("fingerprint", "corpus or question-file fingerprints"), ("split", "splits"),
                           ("match", "match modes")):
            if a[key] != b[key]:
                raise ValueError(f"refusing to compare runs with different {label}; use --force to override")
    ranks_a = {q["id"]: q["rank"] for q in a["questions"] if q["category"] != "unanswerable"}
    ranks_b = {q["id"]: q["rank"] for q in b["questions"] if q["category"] != "unanswerable"}
    ids = sorted(ranks_a.keys() & ranks_b.keys())

    def within(rank, k):
        return rank is not None and rank <= k

    out = {"questions": len(ids), "changed_ranks": sum(ranks_a[i] != ranks_b[i] for i in ids)}
    for k in (10, 3):
        out[f"gained@{k}"] = [i for i in ids if within(ranks_b[i], k) and not within(ranks_a[i], k)]
        out[f"lost@{k}"] = [i for i in ids if within(ranks_a[i], k) and not within(ranks_b[i], k)]
        out[f"net@{k}"] = len(out[f"gained@{k}"]) - len(out[f"lost@{k}"])
    overall_a, overall_b = a["metrics"]["overall"], b["metrics"]["overall"]
    out["deltas"] = {m: overall_b[m]["value"] - overall_a[m]["value"] for m in metric_names()}
    net, mrr = out["net@10"], out["deltas"]["mrr@10"]
    eps = 1e-12
    if net >= MIN_NET_QUESTIONS and mrr >= -eps:
        out["verdict"] = "improved"
    elif net <= -MIN_NET_QUESTIONS and mrr <= eps:
        out["verdict"] = "worse"
    elif abs(net) >= MIN_NET_QUESTIONS:
        out["verdict"] = "mixed"
    else:
        out["verdict"] = "no measurable change"
    return out


def format_comparison(c: dict) -> str:
    deltas = ", ".join(f"{m} {d:+.3f}" for m, d in c["deltas"].items())
    return "\n".join([
        f"{c['questions']} paired questions, {c['changed_ranks']} with a different rank", f"Deltas: {deltas}",
        f"Rank 10: gained {len(c['gained@10'])} {c['gained@10']}, lost {len(c['lost@10'])} {c['lost@10']}",
        f"Rank 3: gained {len(c['gained@3'])} {c['gained@3']}, lost {len(c['lost@3'])} {c['lost@3']}",
        f"Verdict: {c['verdict']} (needs a net change of at least {MIN_NET_QUESTIONS} questions at rank 10)"])


# ---- lint, audit, selftest ---------------------------------------------------------------------------------------

def book_pages(root: Path, passages: list[Passage]) -> dict[str, int]:
    pages = {}
    for p in passages:
        pages[p.source] = max(pages.get(p.source, 0), p.page_end or p.page or 0)
    for path in (root / "data" / "processed" / "books").glob("*.manifest.json"):
        m = json.loads(path.read_text(encoding="utf-8"))
        pages[m["source"]] = m["pages"]
    return pages


def lint(questions: list[Question], passages: list[Passage], root: Path = ROOT) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    matcher, pages = Matcher(passages, "anchor"), book_pages(root, passages)
    seen = set()
    for q in questions:
        def err(message):
            errors.append(f"{q.id}: {message}")
        if q.id in seen:
            err("duplicate id")
        seen.add(q.id)
        if q.category not in CATEGORIES:
            err(f"unknown category {q.category!r}")
        if q.split not in SPLITS:
            err(f"unknown split {q.split!r}")
        if not q.question.strip():
            err("empty question")
        if q.category == "unanswerable":
            if q.gold:
                err("unanswerable question must have empty gold")
            continue
        if not q.gold:
            err("answerable question has no gold")
        if q.category == "multi_book" and len({g.source for g in q.gold}) < 2:
            err("multi_book question needs gold in at least 2 sources")
        matched: list[Passage] = []
        for g in q.gold:
            if g.source not in matcher.by_source:
                err(f"unknown source {g.source!r}")
                continue
            ok = (len(g.pages) == 2 and all(isinstance(x, int) for x in g.pages) and 1 <= g.pages[0] <= g.pages[1])
            if not ok:
                err(f"pages must be [a, b] with 1 <= a <= b, got {list(g.pages)}")
                continue
            if g.pages[1] > pages.get(g.source, 0):
                err(f"pages {list(g.pages)} are outside {short_source(g.source)} ({pages.get(g.source)} pages)")
            if g.pages[1] - g.pages[0] + 1 > MAX_GOLD_PAGES:
                err(f"pages {list(g.pages)} are wider than {MAX_GOLD_PAGES} pages")
            if not 1 <= len(g.anchor) <= MAX_ANCHORS or any(
                    not 1 <= len(normalize(a).split()) <= MAX_ANCHOR_WORDS for a in g.anchor):
                err(f"needs 1-{MAX_ANCHORS} anchors of 1-{MAX_ANCHOR_WORDS} words")
                continue
            near = [p for p in matcher.by_source[g.source] if matcher.overlaps(p, g)]
            for anchor in g.anchor:
                holders = [p for p in near if normalize(anchor) in matcher.text(p)]
                if not holders:
                    err(f"anchor {anchor!r} occurs in no chunk overlapping pages {list(g.pages)}")
                matched += holders
        if q.category in ("table", "figure") and matched and not any(p.type == q.category for p in matched):
            err(f"{q.category} question, but no anchored chunk has type {q.category}")
        if q.category == "figure" and matched and all(p.credit_only for p in matched):
            err("figure question is answered only by a credit line")
        if matched and q.category in ("paraphrase", "lexical"):
            share = term_share(q.question, matched)
            if q.category == "paraphrase" and share > MAX_PARAPHRASE_OVERLAP:
                err(f"leakage: paraphrase shares {share:.2f} of its terms with the gold chunk "
                    f"(max {MAX_PARAPHRASE_OVERLAP})")
            if q.category == "lexical" and share < MIN_LEXICAL_OVERLAP:
                warnings.append(f"{q.id}: lexical question shares only {share:.2f} of its terms (want >= "
                                f"{MIN_LEXICAL_OVERLAP})")
    counts = Counter(q.category for q in questions)
    for category, minimum in MIN_COUNTS.items():
        if counts[category] < minimum:
            warnings.append(f"only {counts[category]} {category} questions (want at least {minimum})")
    for source in sorted(matcher.by_source):
        entries = [(q, g) for q in questions for g in q.gold if g.source == source and q.category != "unanswerable"]
        if len({q.id for q, _ in entries}) < MIN_BOOK_QUESTIONS:
            warnings.append(f"{short_source(source)} has {len({q.id for q, _ in entries})} answerable questions "
                            f"(want at least {MIN_BOOK_QUESTIONS})")
        early = [q.id for q, g in entries if g.pages[0] <= pages.get(source, 0) / 4]
        if entries and len(set(early)) / len({q.id for q, _ in entries}) > MAX_FIRST_QUARTER_SHARE:
            warnings.append(f"{short_source(source)}: {len(set(early))} of {len({q.id for q, _ in entries})} "
                            f"questions fall in the first quarter of its pages (limit {MAX_FIRST_QUARTER_SHARE:.0%})")
    return errors, warnings


def audit(questions: list[Question], passages: list[Passage], count: int, seed: int) -> str:
    matcher = Matcher(passages, "anchor")
    pool = [q for q in questions if q.gold]
    lines = []
    for q in random.Random(seed).sample(pool, min(count, len(pool))):
        lines.append(f"{q.id} [{q.category}, {q.split}] {q.question}")
        for g in q.gold:
            lines.append(f"    gold: {short_source(g.source)} pages {g.pages[0]}-{g.pages[1]} anchors {list(g.anchor)}")
            chunk = next((p for p in matcher.by_source.get(g.source, ()) if matcher.hit(p, g)), None)
            lines.append("    chunk: " + (chunk.text[:300].replace("\n", " ") if chunk else "(none found)"))
    return "\n".join(lines)


def selftest(passages: list[Passage], sample: int = SELFTEST_SAMPLE, seed: int = SELFTEST_SEED,
             min_words: int = SELFTEST_MIN_WORDS) -> dict:
    """Query with a chunk's own first sentence; its chunk should come back in the top few."""
    index = Index(passages)
    pool = [p for p in passages if p.type == "text" and len(p.text.split()) >= min_words
            and any(len(s.split()) >= 8 for s in re.split(r"(?<=[.!?])\s+", p.text))]
    chosen = random.Random(seed).sample(pool, min(sample, len(pool)))
    per_book: dict[str, list[bool]] = {}
    for p in chosen:
        sentence = next(s for s in re.split(r"(?<=[.!?])\s+", p.text) if len(s.split()) >= 8)
        top = index.search(sentence, SELFTEST_TOP)
        per_book.setdefault(p.source, []).append(any(t is p for t in top))
    flat = [x for hits in per_book.values() for x in hits]
    return {"n": len(flat), "share": sum(flat) / len(flat) if flat else 0.0,
            "per_book": {b: {"n": len(h), "share": sum(h) / len(h)} for b, h in sorted(per_book.items())}}


# ---- command line ------------------------------------------------------------------------------------------------

def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def main(argv: list[str] | None = None) -> int:
    use_utf8()
    parser = argparse.ArgumentParser(description="Evaluate retrieval against the labelled questions in eval/.")
    parser.add_argument("--split", choices=SPLITS, default="dev")
    parser.add_argument("--final", action="store_true", help="required to evaluate the holdout split")
    parser.add_argument("--match", choices=("anchor", "page"), default="anchor")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    parser.add_argument("--save", type=Path, help="write the full result as JSON")
    parser.add_argument("--lint", action="store_true")
    parser.add_argument("--audit", type=int, metavar="N")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    parser.add_argument("--force", action="store_true", help="compare runs even if their fingerprints differ")
    parser.add_argument("--ranking", action="append", metavar="KEY=VALUE",
                        help="override a ranking weight for this run (repeatable); 'neutral' resets all weights")
    parser.add_argument("--sweep", action="append", metavar="KEY=V1,V2,...",
                        help="run one ranking key at several values and print one row per value")
    parser.add_argument("--against", type=Path, help="saved run used as the reference for --sweep (default: neutral)")
    parser.add_argument("--mode", choices=MODES, default=DEFAULT_MODE,
                        help=f"retrieval to evaluate: bm25, dense or hybrid (default: {DEFAULT_MODE})")
    parser.add_argument("--model", choices=sorted(MODELS), default=DEFAULT_MODEL,
                        help="embedding model for dense/hybrid (see models/README.md)")
    parser.add_argument("--lexical-weight", type=float, default=LEXICAL_WEIGHT,
                        help=f"hybrid only: BM25's weight in the rank fusion, dense counts 1.0 (default: {LEXICAL_WEIGHT})")
    args = parser.parse_args(argv)

    if args.compare:
        try:
            runs = [json.loads(_resolve(p).read_text(encoding="utf-8")) for p in args.compare]
            print(format_comparison(compare(*runs, force=args.force)))
        except (ValueError, OSError) as error:
            parser.error(str(error))
        return 0
    if args.split.startswith("holdout") and not args.final:
        parser.error("the holdout splits are for the final report only; pass --final to use one")
    if args.split.startswith("holdout"):
        print("note: holdout is for the final report only; do not tune on it")

    passages = load_corpus(base=ROOT)
    if not passages:
        parser.error("no corpus found; run rag-ingest first")
    if args.selftest:
        result = selftest(passages)
        print(f"self-retrieval in the top {SELFTEST_TOP}: {result['share']:.1%} of {result['n']} chunks")
        for book, s in result["per_book"].items():
            print(f"  {short_source(book):<34}{s['share']:.1%} of {s['n']}")
        return 0 if result["share"] >= SELFTEST_MIN_SHARE else 1
    questions_path = _resolve(args.questions)
    try:
        questions = load_questions(questions_path)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.lint:
        errors, warnings = lint(questions, passages)
        for message in errors:
            print("ERROR:", message)
        for message in warnings:
            print("WARNING:", message)
        print(f"{len(questions)} questions: {len(errors)} errors, {len(warnings)} warnings")
        return 1 if errors else 0
    if args.audit:
        print(audit(questions, passages, args.audit, args.seed))
        return 0
    try:
        ranking = parse_ranking(args.ranking)
    except ValueError as error:
        parser.error(str(error))
    try:
        index = build_index(passages, args.mode, model=args.model, lexical_weight=args.lexical_weight)
    except (ModelMissing, VectorsMissing) as error:
        parser.error(f"{error} (or pass --mode bm25)")
    if args.sweep:
        if args.save:
            parser.error("--save cannot be combined with --sweep")
        try:
            reference = (json.loads(_resolve(args.against).read_text(encoding="utf-8")) if args.against else
                         evaluate(index, questions, passages, questions_path, args.split, args.match, ranking=NEUTRAL))
            for spec in args.sweep:
                key, _, values = spec.partition("=")
                print("\n".join(sweep(index, questions, passages, questions_path, args.split, args.match, ranking,
                                      key, [v for v in values.split(",") if v], reference)))
        except (ValueError, OSError) as error:
            parser.error(str(error))
        return 0
    result = evaluate(index, questions, passages, questions_path, args.split, args.match, ranking=ranking)
    print(format_report(result))
    if args.save:
        _resolve(args.save).parent.mkdir(parents=True, exist_ok=True)
        _resolve(args.save).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved {_resolve(args.save)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
