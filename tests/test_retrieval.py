import json
import math
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from rag_mvp.credits import is_credit_line
from rag_mvp.retrieval import NEUTRAL, Index, Passage, RankingConfig, load_passages, retrieve, tokenize


class Tokenising(unittest.TestCase):
    def test_stopwords_accents_and_plurals(self):
        self.assertEqual(tokenize("What are the fishes of H\u00f6\u00f6k?"), ["fish", "hook"])
        self.assertEqual(tokenize("studies"), tokenize("study"))
        self.assertEqual(tokenize("Gills"), tokenize("gill"))


class Ranking(unittest.TestCase):
    def test_rare_term_beats_a_passage_stuffed_with_common_words(self):
        passages = [
            Passage("a", 1, "what is the of in a and the what is the of " * 10 + "reef"),
            Passage("b", 1, "The lagoon is the calm water behind the reef crest."),
        ]
        self.assertEqual(retrieve("what is the role of the lagoon", passages, 1)[0].source, "b")

    def test_idf_prefers_the_distinctive_term(self):
        passages = [Passage("common", 1, "fish fish fish fish swim in water"),
                    Passage("rare", 1, "the protogyny of wrasse is unusual among fish")]
        self.assertEqual(retrieve("fish protogyny", passages, 1)[0].source, "rare")

    def test_short_focused_passage_beats_a_long_one_with_one_mention(self):
        passages = [Passage("long", 1, "osmoregulation " + "unrelated words about plankton " * 80),
                    Passage("short", 1, "Osmoregulation in marine teleosts depends on the gills.")]
        self.assertEqual(retrieve("osmoregulation teleosts gills", passages, 1)[0].source, "short")

    def test_section_titles_are_searchable(self):
        passages = [Passage("a", 1, "Salt is excreted by chloride cells.", section_path=("Osmoregulation",)),
                    Passage("b", 1, "Salt is mined in large quantities.", section_path=("Mining",))]
        self.assertEqual(retrieve("osmoregulation", passages, 1)[0].source, "a")

    def test_type_filter_and_no_match(self):
        passages = [Passage("t", 1, "mass of parrotfish", type="table"),
                    Passage("p", 1, "parrotfish eat algae", type="text")]
        self.assertEqual([m.source for m in retrieve("parrotfish", passages, 5, {"table"})], ["t"])
        self.assertEqual(retrieve("unicorn", passages), [])
        self.assertEqual(retrieve("the of and", passages), [])  # only stopwords

    def test_index_can_be_reused_and_is_stable_on_ties(self):
        passages = [Passage("x", i, "coral reef") for i in range(1, 4)]
        index = Index(passages)
        self.assertEqual([m.index for m in index.search("coral", 3)], [1, 2, 3])
        self.assertEqual(index.search("coral", 1)[0].index, 1)


class RankingWeights(unittest.TestCase):
    def pair(self, flagged_kind="low_quality"):
        """Two passages with identical text: the flagged one comes first, so only a weight can demote it."""
        kind = {"low_quality": dict(low_quality=True), "table_degraded": dict(table_degraded=True, type="table")}
        flagged = Passage("flagged", 1, "coral reef lagoon fish", **kind[flagged_kind])
        clean = Passage("clean", 2, "coral reef lagoon fish", type=flagged.type)
        return [flagged, clean]

    def test_neutral_scoring_is_exactly_plain_bm25(self):
        passages = [Passage("s", i, f"coral reef " * (i % 4 + 1) + "fish lagoon " * (i % 3) + f"word{i}",
                            type=("figure", "table", "text")[i % 3], low_quality=i % 5 == 0, table_degraded=i % 7 == 0)
                    for i in range(1, 40)]
        index = Index(passages)
        n, scores = len(passages), {}
        for term in set(tokenize("coral fish lagoon")):
            postings = index.postings.get(term, ())
            idf = math.log(1 + (n - len(postings) + 0.5) / (len(postings) + 0.5))
            for position, count in postings:
                norm = count + 1.5 * (1 - 0.75 + 0.75 * index.lengths[position] / index.average)
                scores[position] = scores.get(position, 0.0) + idf * count * 2.5 / norm
        expected = [(passages[p], scores[p]) for p in sorted(scores, key=lambda p: (-scores[p], p))]
        self.assertEqual(index.search_scored("coral fish lagoon", 100), expected)
        self.assertEqual(index.search_scored("coral fish lagoon", 100, ranking=NEUTRAL), expected)

    def test_each_weight_alone_moves_a_flagged_passage_below_an_equal_clean_one(self):
        for kind, weight in (("low_quality", "low_quality_weight"), ("table_degraded", "table_degraded_weight")):
            with self.subTest(kind):
                index = Index(self.pair(kind))
                self.assertEqual([p.source for p in index.search("coral", 2)], ["flagged", "clean"])
                demoted = replace(RankingConfig(), **{weight: 0.9})
                self.assertEqual([p.source for p in index.search("coral", 2, ranking=demoted)], ["clean", "flagged"])
        index = Index([Passage("fig", 1, "coral reef", type="figure"), Passage("txt", 2, "coral reef")])
        weighted = RankingConfig(type_weights={"figure": 0.9})
        self.assertEqual([p.source for p in index.search("coral", 2, ranking=weighted)], ["txt", "fig"])

    def test_a_zero_weight_hides_the_passage(self):
        index = Index(self.pair())
        hidden = RankingConfig(low_quality_weight=0)
        self.assertEqual([p.source for p in index.search("coral", 5, ranking=hidden)], ["clean"])

    def test_weights_must_be_non_negative_numbers(self):
        for bad in (dict(low_quality_weight=-0.1), dict(credit_only_weight="x"), dict(type_weights={"figure": -1}),
                    dict(table_degraded_weight=True)):
            with self.subTest(bad), self.assertRaises(ValueError):
                RankingConfig(**bad)

    def test_a_per_call_override_does_not_change_the_index_default(self):
        index = Index(self.pair())
        index.search("coral", 2, ranking=RankingConfig(low_quality_weight=0.5))
        self.assertEqual(index.ranking, RankingConfig())
        self.assertEqual([p.source for p in index.search("coral", 2)], ["flagged", "clean"])
        self.assertEqual(Index(self.pair(), ranking=RankingConfig(low_quality_weight=0.5)).ranking.low_quality_weight,
                         0.5)

    def test_a_uniform_figure_weight_does_not_reorder_a_figure_only_search(self):
        passages = [Passage("f", i, "coral " * i + "reef", type="figure") for i in range(1, 5)]
        index = Index(passages)
        plain = index.search("coral", 4, {"figure"})
        weighted = index.search("coral", 4, {"figure"}, RankingConfig(type_weights={"figure": 0.3}))
        self.assertEqual(plain, weighted)


class CreditLines(unittest.TestCase):
    def test_word_boundaries_keep_real_captions_out(self):
        for text in ("Photosynthesis in kelp", "Credits and acknowledgements", "Courtesies of the reef",
                     "Figure 3.2 Kelp forest", "A photograph of the reef crest"):
            with self.subTest(text):
                self.assertFalse(is_credit_line(text))
        for text in ("Photo by A. Smith", "Courtesy of NOAA", "\u00a9 2019 Jane Doe", "Source: FAO",
                     "\u201cTurks and Caicos Glow Worms 2017\u201d by Alana Olendorf [CC by 4.0]",
                     "Squirrelfish by Skinned Mink via Flickr", "Reef [CC SA 3.0]", "Fish, public domain"):
            with self.subTest(text):
                self.assertTrue(is_credit_line(text))

    def load(self, *records):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "b.jsonl").write_text("\n".join(json.dumps({"source": "b.pdf", **r}) for r in records),
                                          encoding="utf-8")
            return load_passages(root)

    def test_credit_only_needs_a_short_figure_chunk_that_is_a_credit(self):
        long_credit = "Photo by A. Smith " + "reef " * 14
        marked = [p.credit_only for p in self.load(
            {"type": "figure", "text": "Photo by A. Smith [CC by 2.0]", "images": ["d/p1.jpg"]},
            {"type": "figure", "text": "Photosynthesis in kelp"},
            {"type": "figure", "text": "Figure 3.2 Kelp forest"},
            {"type": "figure", "text": "A photo of the reef crest taken at dawn by the dive team"},
            {"type": "figure", "text": long_credit},
            {"type": "text", "text": "Photo by A. Smith"})]
        self.assertEqual(marked, [True, False, False, False, False, False])

    def test_a_credit_only_passage_is_hidden_at_weight_zero_and_still_reachable_otherwise(self):
        credit = Passage("c", 1, "Photo by A. Smith of a coral", type="figure", images=("d/p1.jpg",), credit_only=True)
        caption = Passage("d", 2, "Diagram of a coral polyp", type="figure")
        index = Index([credit, caption])
        self.assertEqual([p.source for p in index.search("coral", 5)], ["c", "d"])
        self.assertEqual(index.search("coral", 5)[0].images, ("d/p1.jpg",))
        self.assertEqual([p.source for p in index.search("coral", 5, ranking=RankingConfig(credit_only_weight=0))],
                         ["d"])
        demoted = RankingConfig(credit_only_weight=0.5)
        self.assertEqual([p.source for p in index.search("coral", 5, ranking=demoted)], ["d", "c"])

    def test_search_does_not_import_the_pdf_library(self):
        code = "import sys, rag_mvp.retrieval, rag_mvp.cli; print('pymupdf' in sys.modules)"
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual((done.returncode, done.stdout.strip()), (0, "False"), done.stderr)


class Loading(unittest.TestCase):
    def test_quality_metadata_survives_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "book.jsonl").write_text(
                '{"id": "d-2", "source": "b.pdf", "text": "x", "chapter": "Ch 1", "page_label": 30, '
                '"page_label_end": 31, "low_quality": true, "table_structure_degraded": true}\n', encoding="utf-8")
            p = load_passages(root)[0]
        self.assertEqual((p.chapter, p.page_label_end, p.low_quality, p.table_degraded), ("Ch 1", 31, True, True))
        clean = Passage("a", 1, "x")
        self.assertEqual((clean.chapter, clean.page_label_end, clean.low_quality, clean.table_degraded),
                         ("", None, False, False))
    def test_loads_chunk_records_with_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "book.jsonl").write_text(
                '{"id": "d-1", "source": "b.pdf", "type": "table", "text": "| a |", "page": 4, "page_end": 5, '
                '"page_label": 2, "section_path": ["Ch 1", "Sec"], "images": ["d/p1.jpg"]}\n', encoding="utf-8")
            (root / "book.manifest.json").write_text("{}", encoding="utf-8")
            passages = load_passages(root)
        self.assertEqual(len(passages), 1)
        p = passages[0]
        self.assertEqual((p.id, p.type, p.page, p.page_end, p.page_label, p.section_path, p.images),
                         ("d-1", "table", 4, 5, 2, ("Ch 1", "Sec"), ("d/p1.jpg",)))

    def test_text_files_are_windowed_at_whitespace_not_mid_word(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "nested" / "notes.md").write_text("alpha beta gamma delta epsilon zeta eta theta", encoding="utf-8")
            (root / "ignore.csv").write_text("alpha", encoding="utf-8")
            passages = load_passages(root, chunk_size=20, overlap=6)
        self.assertTrue(all(p.source == "nested/notes.md" for p in passages))
        words = set("alpha beta gamma delta epsilon zeta eta theta".split())
        for passage in passages:
            self.assertLessEqual(len(passage.text), 20)
            self.assertTrue(set(passage.text.split()) <= words, passage.text)
        self.assertIn("theta", passages[-1].text)


if __name__ == "__main__":
    unittest.main()
