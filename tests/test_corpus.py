"""Checks on the REAL ingested textbooks. Skipped when data/processed/books is empty; run rag-ingest first.

These guard against regressions that clean synthetic PDFs cannot show: real page furniture, real OCR layers,
real two-column layouts. Thresholds sit just outside measured values, so a real regression fails them."""

import hashlib
import json
import re
import unittest
from collections import Counter
from pathlib import Path

from rag_mvp.evaluation import MIN_COUNTS, SELFTEST_MIN_SHARE, SPLITS, lint, load_questions, selftest
from rag_mvp.ingest import EXTRACTOR_VERSION, MANIFEST_SCHEMA, excluded_share
from rag_mvp.ingest_config import DetectConfig, load_settings
from rag_mvp.retrieval import Index, load_corpus

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "ingest_config.json"
QUESTIONS = ROOT / "eval" / "questions.jsonl"
BOOKS = ROOT / "data" / "processed" / "books"
FIGURES = ROOT / "data" / "processed" / "figures"


def load_books() -> dict[str, tuple[dict, list[dict]]]:
    books = {}
    for manifest_path in sorted(BOOKS.glob("*.manifest.json")):
        jsonl = manifest_path.with_name(manifest_path.name.replace(".manifest.json", ".jsonl"))
        if not jsonl.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunks = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
        books[manifest["source"]] = (manifest, chunks)
    return books


BOOKS_DATA = load_books()


@unittest.skipUnless(BOOKS_DATA and QUESTIONS.exists(), "needs ingested books and eval/questions.jsonl")
class RealEvaluationSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.passages = load_corpus(base=ROOT)
        cls.questions = load_questions(QUESTIONS)

    def test_questions_pass_the_lint_against_the_real_corpus(self):
        errors, _ = lint(self.questions, self.passages, ROOT)
        self.assertEqual(errors, [])

    def test_composition_meets_the_minimums_and_every_source_and_split_is_valid(self):
        counts = Counter(q.category for q in self.questions)
        for category, minimum in MIN_COUNTS.items():
            self.assertGreaterEqual(counts[category], minimum, category)
        self.assertGreaterEqual(len(self.questions), 96)
        self.assertTrue(all(q.split in SPLITS for q in self.questions))
        self.assertTrue({g.source for q in self.questions for g in q.gold} <= set(BOOKS_DATA))
        self.assertTrue(any(q.split == "holdout2" for q in self.questions))

    def test_the_self_test_passes_overall(self):
        result = selftest(self.passages)
        self.assertGreaterEqual(result["share"], SELFTEST_MIN_SHARE, result["per_book"])

    def test_loading_and_searching_leave_the_chunk_files_untouched(self):
        files = sorted(BOOKS.glob("*.jsonl"))
        before = [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]
        Index(load_corpus(base=ROOT)).search("coral reef", 5)
        self.assertEqual([hashlib.sha256(f.read_bytes()).hexdigest() for f in files], before)


@unittest.skipUnless(BOOKS_DATA, "no ingested books in data/processed/books; run rag-ingest first")
class RealBookInvariants(unittest.TestCase):
    def books(self):
        return [(source[:40], manifest, chunks) for source, (manifest, chunks) in BOOKS_DATA.items()]

    def test_every_book_was_ingested_with_the_current_extractor(self):
        for name, manifest, _ in self.books():
            with self.subTest(book=name):
                self.assertEqual(manifest["extractor_version"], EXTRACTOR_VERSION, "re-run rag-ingest")

    def test_every_manifest_has_the_current_schema(self):
        for name, manifest, _ in self.books():
            with self.subTest(book=name):
                self.assertEqual(manifest.get("manifest_schema", 1), MANIFEST_SCHEMA,
                                 "run rag-ingest --upgrade-manifests")

    def test_no_book_has_an_unreviewed_suspect_page_run(self):
        for name, manifest, _ in self.books():
            with self.subTest(book=name):
                self.assertEqual([r for r in manifest["suspect_page_runs"] if not r["reviewed"]], [])

    def test_no_book_is_mostly_excluded_or_has_an_overlong_outline_section(self):
        defaults, configs = load_settings(CONFIG) if CONFIG.exists() else (DetectConfig(), {})
        for name, manifest, _ in self.books():
            detect = configs[manifest["source"]].detect if manifest["source"] in configs else defaults
            with self.subTest(book=name):
                share = excluded_share({int(p): r for p, r in manifest["excluded_pages"].items()}, manifest["pages"])
                self.assertLessEqual(share, detect.max_excluded_share)
                self.assertEqual(manifest["long_reference_sections"], [])

    def test_every_config_entry_refers_to_a_book_that_exists_with_the_same_hash(self):
        if not CONFIG.exists():
            self.skipTest("no ingest_config.json")
        digests = {m["source"]: m["sha256"] for m, _ in BOOKS_DATA.values()}
        for name, entry in load_settings(CONFIG)[1].items():
            with self.subTest(book=name):
                pdf = ROOT / "data" / "raw" / name
                self.assertTrue(name in digests or pdf.exists(), "config names a book that is not in the corpus")
                if entry.sha256:
                    actual = digests.get(name) or hashlib.sha256(pdf.read_bytes()).hexdigest()
                    self.assertEqual(entry.sha256, actual)

    def test_no_book_specific_strings_in_the_source_code(self):
        stems = {Path(source).stem for source in BOOKS_DATA}
        stems |= {p.stem for p in (ROOT / "data" / "raw").glob("*.pdf")}
        if CONFIG.exists():
            stems |= {Path(name).stem for name in load_settings(CONFIG)[1]}
        code = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "src" / "rag_mvp").glob("*.py"))
        for stem in sorted(stems):
            with self.subTest(book=stem[:40]):
                self.assertNotIn(stem, code)

    def test_no_character_level_defects_remain(self):
        for name, _, chunks in self.books():
            with self.subTest(book=name):
                text = " ".join(c["text"] for c in chunks)
                self.assertEqual(text.count("\xad"), 0)
                self.assertEqual(sum(text.count(c) for c in "\ufb00\ufb01\ufb02\ufb03\ufb04"), 0)
                self.assertEqual(text.count("\u2011"), 0)
                self.assertEqual(text.count("\ufffd"), 0)
                # the old extraction left 1,600-3,700 'sys- tem' style breaks per book
                self.assertLess(len(re.findall(r"[a-z]- [a-z]", text)), 80)

    def test_every_chunk_is_citable_and_well_formed(self):
        for name, manifest, chunks in self.books():
            with self.subTest(book=name):
                ids = [c["id"] for c in chunks]
                self.assertEqual(len(ids), len(set(ids)))
                for c in chunks:
                    self.assertIn(c["type"], {"text", "table", "figure"})
                    self.assertTrue(c["text"].strip())
                    self.assertTrue(1 <= c["page"] <= c["page_end"] <= manifest["pages"], c["id"])
                    self.assertEqual(c["doc_id"], manifest["doc_id"])

    def test_text_chunks_are_paragraph_sized_not_page_sized_or_tiny(self):
        for name, _, chunks in self.books():
            with self.subTest(book=name):
                words = sorted(len(c["text"].split()) for c in chunks if c["type"] == "text")
                self.assertGreaterEqual(words[0], 8)
                self.assertLessEqual(words[-1], 420)
                self.assertTrue(100 <= words[len(words) // 2] <= 260, words[len(words) // 2])

    def test_nearly_all_chunks_have_chapter_context(self):
        for name, _, chunks in self.books():
            with self.subTest(book=name):
                missing = [c["page"] for c in chunks if not c["chapter"]]
                self.assertLess(len(missing) / len(chunks), 0.03, sorted(set(missing))[:20])

    def test_front_and_back_matter_is_excluded_not_indexed(self):
        for name, manifest, chunks in self.books():
            with self.subTest(book=name):
                excluded = {int(p) for p in manifest["excluded_pages"]}
                self.assertGreater(len(excluded), 0)
                self.assertEqual([c["id"] for c in chunks if c["page"] in excluded and c["page_end"] in excluded], [])

    def test_manifest_matches_the_pdf_that_was_ingested(self):
        for name, manifest, _ in self.books():
            pdf = ROOT / "data" / "raw" / manifest["source"]
            if pdf.exists():
                with self.subTest(book=name):
                    self.assertEqual(manifest["sha256"], hashlib.sha256(pdf.read_bytes()).hexdigest(),
                                     "PDF changed since ingestion; re-run rag-ingest")

    def test_figure_chunks_point_to_existing_image_files(self):
        if not FIGURES.exists():
            self.skipTest("figures were not rendered (--no-figures)")
        for name, _, chunks in self.books():
            with self.subTest(book=name):
                images = [i for c in chunks for i in c["images"]]
                self.assertGreater(len(images), 0)
                for image in images:
                    self.assertGreater((FIGURES / image).stat().st_size, 1000, image)

    def test_most_captioned_figures_have_an_image(self):
        for name, _, chunks in self.books():
            figures = [c for c in chunks if c["type"] == "figure"]
            if len(figures) >= 20:
                with self.subTest(book=name):
                    with_image = sum(1 for c in figures if c["images"])
                    self.assertGreater(with_image / len(figures), 0.75, f"{with_image}/{len(figures)}")

    def test_unreliable_text_is_flagged_and_rare(self):
        for name, manifest, chunks in self.books():
            with self.subTest(book=name):
                flagged = sum(1 for c in chunks if c["low_quality"])
                self.assertLess(flagged / len(chunks), 0.03, f"{flagged}/{len(chunks)}")
                self.assertEqual(manifest["poor_quality_pages"], sorted(manifest["poor_quality_pages"]))

    def test_tables_with_fused_cells_are_rare_or_flagged(self):
        for name, _, chunks in self.books():
            tables = [c for c in chunks if c["type"] == "table"]
            if not tables:
                continue
            with self.subTest(book=name):
                fused = [c for c in tables if not c["table_structure_degraded"] and max(
                    (len(x.strip()) for row in c["text"].splitlines() if row.startswith("|") for x in row.split("|")),
                    default=0) > 250]
                self.assertEqual([c["page"] for c in fused], [])


if __name__ == "__main__":
    unittest.main()
