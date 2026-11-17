import hashlib
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import pymupdf

import fixtures
from rag_mvp.ingest import (MANIFEST_SCHEMA, excluded_share, find_suspect_runs, ingest_pdf, long_reference_sections,
                            main as ingest_main)
from rag_mvp.ingest_config import BookConfig, DetectConfig, config_for, load_config
from rag_mvp.layout import extract_page
from test_layout_ingest import IngestCase

NORMAL, ODD = (612, 792), (400, 520)
NEW_KEYS = ("manifest_schema", "config_excluded_ranges", "config_accepted_ranges", "suspect_page_runs",
            "long_reference_sections", "excluded_by_reason")


def layout(normal=30, odd=12, position="end", odd_size=ODD):
    sizes = [NORMAL] * normal
    at = {"start": 0, "middle": normal // 2, "end": normal}[position]
    sizes[at:at] = [odd_size] * odd
    numbers = list(range(1, len(sizes) + 1))
    return dict(zip(numbers, sizes)), numbers


class SuspectRuns(unittest.TestCase):
    detect = DetectConfig()

    def runs(self, sizes, numbers, excluded=None, accepted=frozenset(), detect=None):
        return find_suspect_runs(sizes, numbers, excluded or {}, set(accepted), detect or self.detect)

    def test_a_run_of_exactly_the_minimum_is_flagged_and_one_page_fewer_is_not(self):
        runs = self.runs(*layout(odd=10))
        self.assertEqual([(r.first, r.last, r.status, r.reviewed) for r in runs], [(31, 40, "indexed", False)])
        self.assertEqual(self.runs(*layout(odd=9)), [])

    def test_size_tolerance_boundary(self):
        just_inside, just_over = (630, 792), (631, 792)  # 3% of 612 is 18.36
        self.assertEqual(self.runs(*layout(odd_size=just_inside)), [])
        self.assertEqual(len(self.runs(*layout(odd_size=just_over))), 1)

    def test_height_difference_alone_is_enough(self):
        self.assertEqual(len(self.runs(*layout(odd_size=(612, 700)))), 1)

    def test_an_isolated_odd_page_and_a_run_in_the_middle(self):
        sizes, numbers = layout(odd=1)
        self.assertEqual(self.runs(sizes, numbers), [])
        run, = self.runs(*layout(odd=12, position="middle"))
        self.assertEqual((run.first, run.last, (run.width, run.height)), (16, 27, ODD))

    def test_the_dominant_size_does_not_depend_on_page_order(self):
        sizes = {n: NORMAL if n <= 15 else ODD for n in range(1, 31)}
        forward = self.runs(sizes, list(range(1, 31)))
        backward = self.runs(sizes, list(range(30, 0, -1)))
        self.assertEqual(forward, backward)

    def test_pages_excluded_by_the_outline_are_still_found_but_not_reviewed(self):
        sizes, numbers = layout()
        excluded = {n: "outline: Index" for n in range(31, 43)}
        run, = self.runs(sizes, numbers, excluded)
        self.assertEqual((run.status, run.reviewed), ("outline", False))

    def test_config_and_accept_ranges_are_reviewed(self):
        sizes, numbers = layout()
        run, = self.runs(sizes, numbers, {n: "config: foreign book" for n in range(31, 43)})
        self.assertEqual((run.status, run.reviewed), ("config", True))
        run, = self.runs(sizes, numbers, accepted=range(31, 43))
        self.assertEqual((run.status, run.reviewed), ("accepted", True))

    def test_a_mixed_run_is_partial_and_not_reviewed(self):
        sizes, numbers = layout()
        run, = self.runs(sizes, numbers, {n: "config: x" for n in range(31, 37)})
        self.assertEqual((run.status, run.reviewed), ("partial", False))


class ReferenceSections(unittest.TestCase):
    def test_threshold_boundary(self):
        detect = DetectConfig(max_reference_pages=40)
        forty = {n: "outline: Index" for n in range(1, 41)}
        self.assertEqual(long_reference_sections(forty, detect), [])
        forty[41] = "outline: Index"
        found, = long_reference_sections(forty, detect)
        self.assertEqual((found["first"], found["last"], found["pages"]), (1, 41, 41))

    def test_a_section_the_config_took_over_is_not_reported(self):
        excluded = {n: "outline: Index" for n in range(1, 60)}
        excluded.update({n: "config: appended book" for n in range(10, 60)})
        self.assertEqual(long_reference_sections(excluded, DetectConfig()), [])

    def test_different_sections_are_not_merged(self):
        excluded = {n: "outline: Preface" for n in range(1, 30)} | {n: "outline: Index" for n in range(30, 60)}
        self.assertEqual(long_reference_sections(excluded, DetectConfig()), [])

    def test_excluded_share_ignores_config_exclusions(self):
        excluded = {1: "outline: Cover", 2: "outline: Cover", 3: "config: x", 4: "config: x"}
        self.assertEqual(excluded_share(excluded, 10), 0.2)


class ConfigFile(unittest.TestCase):
    def load(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "c.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            return load_config(path)

    def book(self, **entry):
        return {"books": {"a.pdf": entry}}

    def test_a_valid_file_applies_defaults_and_per_book_overrides(self):
        configs = self.load({"defaults": {"detect": {"min_run_pages": 7}}, "books": {
            "a.pdf": {"exclude_pages": [{"pages": [3, 5], "reason": "junk"}], "detect": {"size_tolerance": 0.1}}}})
        a = configs["a.pdf"]
        self.assertEqual((a.exclude, a.detect.min_run_pages, a.detect.size_tolerance), (((3, 5, "junk"),), 7, 0.1))

    def test_each_kind_of_mistake_is_reported_clearly(self):
        bad = {
            "unknown top-level key": ({"bookz": {}}, "unknown key"),
            "unknown book key": (self.book(exclude=[]), "unknown key"),
            "bad range": (self.book(exclude_pages=[{"pages": [5, 3], "reason": "x"}]), "first, last"),
            "zero page": (self.book(exclude_pages=[{"pages": [0, 3], "reason": "x"}]), "first, last"),
            "empty reason": (self.book(exclude_pages=[{"pages": [1, 3], "reason": " "}]), "reason"),
            "missing reason": (self.book(accept_pages=[{"pages": [1, 3]}]), "reason"),
            "bad threshold": (self.book(detect={"min_run_pages": 0}), "positive"),
            "negative threshold": ({"defaults": {"detect": {"size_tolerance": -1}}}, "positive"),
            "typo in detect": (self.book(detect={"min_run_page": 5}), "unknown key"),
        }
        for label, (data, message) in bad.items():
            with self.subTest(label), self.assertRaisesRegex(ValueError, message):
                self.load(data)

    def test_lookup_is_by_name_and_a_hash_mismatch_ignores_the_entry_with_a_warning(self):
        entry = BookConfig("a.pdf", sha256="a" * 64, exclude=((1, 2, "x"),))
        configs = {"a.pdf": entry}
        self.assertEqual(config_for(configs, "a.pdf", "A" * 64), (entry, []))
        self.assertEqual(config_for(configs, "other.pdf", "a" * 64), (None, []))
        found, warnings = config_for(configs, "a.pdf", "b" * 64)
        self.assertIsNone(found)
        self.assertIn("sha256", warnings[0])


class ConfiguredIngest(IngestCase):
    def test_config_excluded_pages_have_no_chunks_and_are_never_ocr_d(self):
        text = "Coral reefs support many fish species and protect the shore from storm waves each year. " * 3
        source = pymupdf.open()
        sheet = source.new_page(width=612, height=792)
        sheet.insert_textbox(pymupdf.Rect(72, 100, 540, 400), text, fontsize=10)
        self.addCleanup(source.close)
        calls = []

        def fake_ocr(page, number):
            calls.append(number)
            return extract_page(sheet, number)

        config = BookConfig("book.pdf", exclude=((3, 4, "appended junk"),))
        _, result, chunks, manifest = self.ingest(fixtures.scanned_book, "", 4, False, ocr=True, ocr_page=fake_ocr,
                                                  book_config=config)
        self.assertEqual(calls, [1, 2])
        self.assertEqual(max(c["page_end"] for c in chunks), 2)
        info = {p["page"]: p["excluded"] for p in manifest["page_info"]}
        self.assertEqual((info[2], info[3], info[4]), (None, "config: appended junk", "config: appended junk"))
        self.assertEqual(manifest["config_excluded_ranges"], [{"pages": [3, 4], "reason": "appended junk"}])

    def test_config_wins_over_an_outline_reason(self):
        config = BookConfig("book.pdf", exclude=((3, 3, "bad page"), (8, 8, "stub")))
        _, result, chunks, manifest = self.ingest(fixtures.book_with_columns, book_config=config)
        self.assertEqual(result.excluded_pages[8], "config: stub")
        self.assertEqual(result.excluded_pages[1], "outline: Contents")
        self.assertNotIn(3, {c["page"] for c in chunks})
        self.assertIn(4, {c["page"] for c in chunks})
        self.assertEqual(manifest["manifest_schema"], MANIFEST_SCHEMA)

    def test_a_configured_page_outside_the_book_is_an_error(self):
        config = BookConfig("book.pdf", exclude=((7, 99, "typo"),))
        with self.assertRaisesRegex(ValueError, r"book\.pdf.*99"):
            self.ingest(fixtures.book_with_columns, book_config=config)

    def test_config_is_scoped_to_the_named_book(self):
        out = self.root / "out"
        pdfs = []
        for name in ("a.pdf", "b.pdf"):
            pdf = self.root / name
            fixtures.book_with_columns(pdf)
            pdfs.append(pdf)
        digest = hashlib.sha256(pdfs[0].read_bytes()).hexdigest()
        config = self.root / "c.json"
        config.write_text(json.dumps({"books": {"a.pdf": {"sha256": digest, "exclude_pages": [
            {"pages": [3, 4], "reason": "only for a"}]}}}), encoding="utf-8")
        with redirect_stdout(StringIO()):
            ingest_main([*map(str, pdfs), "--output-dir", str(out), "--no-figures", "--config", str(config)])
        a, b = (json.loads((out / f"{n}.manifest.json").read_text(encoding="utf-8")) for n in ("a", "b"))
        self.assertEqual(a["config_excluded_ranges"], [{"pages": [3, 4], "reason": "only for a"}])
        self.assertEqual(b["config_excluded_ranges"], [])
        self.assertGreater(b["chunks"], a["chunks"])
        self.assertEqual([p["page"] for p in b["page_info"] if (p["excluded"] or "").startswith("config")], [])


class DetectionInIngest(IngestCase):
    def test_a_foreign_sized_run_is_an_unreviewed_warning_and_one_odd_page_is_not(self):
        _, result, _, manifest = self.ingest(fixtures.book_with_odd_pages)
        run, = result.suspect_runs
        self.assertEqual((run.first, run.last, run.status, run.reviewed), (21, 32, "indexed", False))
        self.assertEqual(manifest["suspect_page_runs"][0]["first"], 21)
        self.assertTrue(any("pages 21-32" in w for w in result.warnings))
        _, result, _, manifest = self.ingest(fixtures.book_with_odd_pages, 20, 1, name="single")
        self.assertEqual((result.suspect_runs, result.warnings, manifest["suspect_page_runs"]), ([], [], []))

    def test_an_accept_entry_makes_the_run_reviewed_and_silent(self):
        config = BookConfig("book.pdf", accept=((21, 32, "photo insert on larger paper"),))
        _, result, chunks, manifest = self.ingest(fixtures.book_with_odd_pages, book_config=config)
        self.assertEqual([(r.status, r.reviewed) for r in result.suspect_runs], [("accepted", True)])
        self.assertEqual(result.warnings, [])
        self.assertIn(32, {c["page_end"] for c in chunks})
        self.assertEqual(manifest["config_accepted_ranges"][0]["pages"], [21, 32])

    def test_a_config_exclusion_of_the_run_is_reviewed_and_removes_the_pages(self):
        config = BookConfig("book.pdf", exclude=((21, 32, "foreign"),))
        _, result, chunks, _ = self.ingest(fixtures.book_with_odd_pages, book_config=config)
        self.assertEqual([(r.status, r.reviewed) for r in result.suspect_runs], [("config", True)])
        self.assertEqual(max(c["page_end"] for c in chunks), 20)


class Main(IngestCase):
    def run_main(self, *argv):
        out, err = StringIO(), StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = ingest_main([*map(str, argv), "--output-dir", str(self.root / "out"), "--no-figures"])
        return code, out.getvalue()

    def in_empty_folder(self):
        previous = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, previous)

    def test_a_missing_default_config_is_only_a_note_and_a_missing_explicit_one_is_an_error(self):
        self.in_empty_folder()
        pdf = self.root / "book.pdf"
        fixtures.book_with_columns(pdf)
        code, out = self.run_main(pdf)
        self.assertEqual(code, 0)
        self.assertIn("note:", out)
        self.assertTrue((self.root / "out" / "book.jsonl").exists())
        with self.assertRaises(SystemExit):
            self.run_main(pdf, "--config", self.root / "nope.json")

    def test_strict_exits_2_only_for_an_unreviewed_run(self):
        self.in_empty_folder()
        odd, single = self.root / "odd.pdf", self.root / "single.pdf"
        fixtures.book_with_odd_pages(odd)
        fixtures.book_with_odd_pages(single, 20, 1)
        code, out = self.run_main(odd, "--strict")
        self.assertEqual(code, 2)
        self.assertIn("WARNING: odd.pdf: pages 21-32", out)
        self.assertEqual(self.run_main(odd)[0], 0)
        self.assertEqual(self.run_main(single, "--strict")[0], 0)

    def test_strict_passes_once_the_run_is_accepted_in_the_config(self):
        pdf = self.root / "odd.pdf"
        fixtures.book_with_odd_pages(pdf)
        config = self.root / "c.json"
        config.write_text(json.dumps({"books": {"odd.pdf": {"accept_pages": [
            {"pages": [21, 32], "reason": "looked at it"}]}}}), encoding="utf-8")
        self.assertEqual(self.run_main(pdf, "--strict", "--config", config)[0], 0)


class UpgradeManifests(IngestCase):
    def setUp(self):
        super().setUp()
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.pdf = self.raw / "book.pdf"
        fixtures.book_with_columns(self.pdf)

    def manifest(self, folder):
        return json.loads((self.root / folder / "book.manifest.json").read_text(encoding="utf-8"))

    def ingest_into(self, folder, config=None):
        ingest_pdf(self.pdf, self.root / folder / "book.jsonl", book_config=config)

    def downgrade(self, folder):
        path = self.root / folder / "book.manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        for key in NEW_KEYS:
            del data[key]
        path.write_text(json.dumps(data, indent=1), encoding="utf-8")

    def upgrade(self, folder, config):
        out = StringIO()
        with redirect_stdout(out):
            code = ingest_main(["--upgrade-manifests", "--raw-dir", str(self.raw), "--output-dir",
                                str(self.root / folder), "--config", str(config)])
        return code, out.getvalue()

    def write_config(self, pages):
        digest = hashlib.sha256(self.pdf.read_bytes()).hexdigest()
        path = self.root / "c.json"
        path.write_text(json.dumps({"books": {"book.pdf": {"sha256": digest, "exclude_pages": [
            {"pages": pages, "reason": "stub"}]}}}), encoding="utf-8")
        return path

    def test_an_upgraded_manifest_equals_a_fresh_one_and_the_chunks_do_not_change(self):
        config = self.write_config([8, 8])
        self.ingest_into("fresh", load_config(config)["book.pdf"])
        self.ingest_into("old")
        self.downgrade("old")
        before = (self.root / "old" / "book.jsonl").read_bytes()
        code, out = self.upgrade("old", config)
        self.assertEqual(code, 0, out)
        fresh, upgraded = self.manifest("fresh"), self.manifest("old")
        for key in (*NEW_KEYS, "excluded_pages", "page_info"):
            self.assertEqual(upgraded[key], fresh[key], key)
        self.assertEqual((self.root / "old" / "book.jsonl").read_bytes(), before)
        self.assertEqual(self.upgrade("old", config)[1].strip(), "book.pdf: manifest already at schema 2")

    def test_the_upgrade_refuses_when_a_config_exclusion_would_drop_existing_chunks(self):
        self.ingest_into("old")
        self.downgrade("old")
        path = self.root / "old" / "book.manifest.json"
        manifest_before, chunks_before = path.read_bytes(), (self.root / "old" / "book.jsonl").read_bytes()
        code, out = self.upgrade("old", self.write_config([3, 3]))
        self.assertEqual(code, 1)
        self.assertIn("re-ingest this book instead", out)
        self.assertEqual(path.read_bytes(), manifest_before)
        self.assertEqual((self.root / "old" / "book.jsonl").read_bytes(), chunks_before)

    def test_a_changed_or_missing_pdf_is_skipped(self):
        self.ingest_into("old")
        self.downgrade("old")
        config = self.write_config([8, 8])
        self.pdf.rename(self.raw / "renamed.pdf")
        code, out = self.upgrade("old", config)
        self.assertEqual(code, 0)
        self.assertIn("skipped, PDF not found", out)
        self.assertNotIn("manifest_schema", self.manifest("old"))


if __name__ == "__main__":
    unittest.main()
