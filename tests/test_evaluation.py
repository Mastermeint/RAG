import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag_mvp import cli, evaluation as ev
from rag_mvp.retrieval import NEUTRAL, Index, Passage, RankingConfig, load_corpus, load_passages, retrieve

SRC = "book.pdf"


def passage(page, text, page_end=None, type="text", source=SRC, index=None, **kw):
    return Passage(source, index or page, text, page, page_end, type=type, **kw)


def gold(*pages, anchor=("alpha",), source=SRC):
    return ev.Gold(source, tuple(pages), tuple(anchor))


def question(id, category="lexical", *golds, text="alpha", split="dev"):
    return ev.Question(id, split, category, text, tuple(golds))


class Metrics(unittest.TestCase):
    def setUp(self):
        # equal-length passages, so BM25 order follows how often "alpha" occurs
        self.passages = [passage(1, "alpha alpha alpha"), passage(2, "alpha alpha beta"),
                         passage(3, "alpha beta gamma", page_end=4), passage(5, "beta gamma delta"),
                         passage(6, "gamma delta epsilon"), passage(7, "delta epsilon zeta")]
        self.index = Index(self.passages)

    def ranks(self, mode, *questions):
        outcomes = ev.run_questions(self.index, list(questions), ev.Matcher(self.passages, mode))
        return [o.rank for o in outcomes]

    def test_hits_at_known_ranks_and_misses(self):
        ranks = self.ranks(
            "anchor",
            question("hit2", "lexical", gold(2, 2, anchor=("alpha beta",))),
            question("hit3", "lexical", gold(3, 3)),
            question("absent", "lexical", gold(9, 9)),
            question("multi", "multi_book", gold(9, 9), gold(2, 2, anchor=("alpha alpha",))),
            question("partial", "lexical", gold(4, 5)),  # passage 3 spans pages 3-4, so it overlaps page 4
            question("nolexeme", "lexical", gold(1, 1, anchor=("zeta",))))
        self.assertEqual(ranks, [2, 3, None, 2, 3, None])

    def test_page_overlap_without_the_anchor_is_a_miss_in_anchor_mode_and_a_hit_in_page_mode(self):
        q = question("q", "lexical", gold(1, 1, anchor=("zeta",)))
        self.assertEqual(self.ranks("anchor", q), [None])
        self.assertEqual(self.ranks("page", q), [1])

    def test_vector_values(self):
        self.assertEqual(ev.vector(1), (1.0, 1.0, 1.0, 1.0))
        self.assertEqual(ev.vector(4), (0.0, 1.0, 1.0, 0.25))
        self.assertEqual(ev.vector(11), (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(ev.vector(None), (0.0, 0.0, 0.0, 0.0))

    def test_unanswerable_questions_are_left_out_of_recall_and_groups_with_few_questions_are_descriptive(self):
        outcomes = ev.run_questions(self.index, [question("a", "lexical", gold(2, 2, anchor=("alpha beta",))),
                                                 question("u", "unanswerable")], ev.Matcher(self.passages))
        groups = ev.group_metrics(outcomes)
        self.assertEqual(groups["overall"]["n"], 1)
        self.assertNotIn("unanswerable", groups["category"])
        self.assertTrue(groups["overall"]["descriptive_only"])
        self.assertEqual(groups["source"][SRC]["n"], 1)

    def test_a_multi_book_question_counts_once_for_every_source(self):
        q = question("m", "multi_book", gold(2, 2, anchor=("alpha beta",)), gold(1, 1, source="other.pdf"))
        outcomes = [ev.Outcome(q, 2, 1.0, [], {})]
        self.assertEqual({s: g["n"] for s, g in ev.group_metrics(outcomes)["source"].items()},
                         {SRC: 1, "other.pdf": 1})

    def test_normalisation_folds_accents_and_respects_word_edges(self):
        self.assertEqual(ev.normalize("H\u00f6\u00f6k's  Fish-Eggs"), " hook s fish eggs ")
        self.assertEqual(ev.normalize("Devil\u2019s Hole"), ev.normalize("Devil's Hole"))
        matcher = ev.Matcher([passage(1, "Some Alphabet soup")])
        self.assertFalse(matcher.anchored(matcher.by_source[SRC][0], gold(1, 1, anchor=("alpha",))))


class Bootstrap(unittest.TestCase):
    vectors = [ev.vector(r) for r in (1, 2, 3, 5, 8, 12, None, 1, 4, None)]

    def test_same_seed_same_interval_and_it_contains_the_estimate(self):
        a, b = ev.bootstrap(self.vectors), ev.bootstrap(self.vectors)
        self.assertEqual(a, b)
        for m, (low, high) in enumerate(a):
            point = sum(v[m] for v in self.vectors) / len(self.vectors)
            self.assertLessEqual(low, point)
            self.assertGreaterEqual(high, point)
        self.assertNotEqual(a, ev.bootstrap(self.vectors, seed=1))


def run_result(ranks, **overrides):
    """A minimal saved run: answerable questions q0..qN with the given ranks."""
    questions = [{"id": f"q{i}", "category": "lexical", "rank": r} for i, r in enumerate(ranks)]
    overall = {m: {"value": 0.0, "ci": [0, 0]} for m in ev.metric_names()}
    vectors = [ev.vector(r) for r in ranks]
    for m, name in enumerate(ev.metric_names()):
        overall[name]["value"] = sum(v[m] for v in vectors) / len(vectors)
    return {"split": "dev", "match": "anchor", "fingerprint": {"corpus": [], "questions_sha256": "x"},
            "metrics": {"overall": overall}, "questions": questions} | overrides


class Comparison(unittest.TestCase):
    base = [None, None, 4, 1, 2, 3, 7, 1]

    def test_a_net_gain_of_one_question_is_not_measurable(self):
        c = ev.compare(run_result(self.base), run_result([2, *self.base[1:]]))
        self.assertEqual((c["net@10"], c["verdict"]), (1, "no measurable change"))

    def test_a_net_gain_of_two_with_mrr_not_lower_is_an_improvement(self):
        c = ev.compare(run_result(self.base), run_result([5, 6, *self.base[2:]]))
        self.assertEqual((c["net@10"], c["verdict"]), (2, "improved"))

    def test_mirror_case_is_worse(self):
        c = ev.compare(run_result([5, 6, *self.base[2:]]), run_result(self.base))
        self.assertEqual(c["verdict"], "worse")

    def test_recall_up_with_mrr_down_is_mixed(self):
        a = run_result([1, 1, 1, 1, None, None])
        b = run_result([3, 3, 3, 3, 9, 9])
        self.assertEqual(ev.compare(a, b)["verdict"], "mixed")

    def test_identical_runs_have_no_changed_ranks(self):
        c = ev.compare(run_result(self.base), run_result(self.base))
        self.assertEqual((c["changed_ranks"], c["verdict"]), (0, "no measurable change"))

    def test_mismatched_fingerprint_split_or_match_is_refused_unless_forced(self):
        a = run_result(self.base)
        for key, value in (("fingerprint", {"corpus": [1], "questions_sha256": "x"}), ("split", "holdout"),
                           ("match", "page")):
            with self.subTest(key), self.assertRaises(ValueError):
                ev.compare(a, run_result(self.base, **{key: value}))
        self.assertEqual(ev.compare(a, run_result(self.base, split="holdout"), force=True)["changed_ranks"], 0)


class Lint(unittest.TestCase):
    def setUp(self):
        self.passages = [
            passage(10, "Urea is retained in the blood of sharks so water does not leave their bodies."),
            passage(11, "Mangrove roots filter salt.", type="figure"),
            passage(12, "| Species | Mass |\n| Grouper | 5400 |", type="table", index=3),
            passage(30, "Other text.", source="second.pdf")]

    def errors(self, *questions):
        with tempfile.TemporaryDirectory() as directory:
            return ev.lint(list(questions), self.passages, Path(directory))

    def assertError(self, text, *questions):
        errors, _ = self.errors(*questions)
        self.assertTrue(any(text in e for e in errors), (text, errors))

    def test_a_clean_question_passes(self):
        errors, _ = self.errors(question("q1", "lexical", gold(10, 10, anchor=("urea",)), text="urea sharks blood"))
        self.assertEqual(errors, [])

    def test_each_error_fires(self):
        ok = gold(10, 10, anchor=("urea",))
        cases = {
            "duplicate id": [question("a", "lexical", ok), question("a", "lexical", ok)],
            "unknown category": [question("a", "weird", ok)],
            "unknown split": [question("a", "lexical", ok, split="test")],
            "unknown source": [question("a", "lexical", gold(10, 10, source="nope.pdf"))],
            "outside": [question("a", "lexical", gold(10, 99, anchor=("urea",)))],
            "wider than": [question("a", "lexical", gold(8, 12, anchor=("urea",)))],
            "pages must be": [question("a", "lexical", gold(12, 10, anchor=("urea",)))],
            "needs 1-3 anchors": [question("a", "lexical", gold(10, 10, anchor=()))],
            "anchors of 1-6 words": [question("a", "lexical", gold(10, 10, anchor=("one two three four five six seven",)))],
            "occurs in no chunk": [question("a", "lexical", gold(10, 10, anchor=("zebra",)))],
            "no anchored chunk has type table": [question("a", "table", gold(10, 10, anchor=("urea",)))],
            "no anchored chunk has type figure": [question("a", "figure", gold(10, 10, anchor=("urea",)))],
            "at least 2 sources": [question("a", "multi_book", ok)],
            "must have empty gold": [question("a", "unanswerable", ok)],
            "has no gold": [question("a", "lexical")],
        }
        for text, questions in cases.items():
            with self.subTest(text):
                self.assertError(text, *questions)

    def test_leakage_check_catches_a_paraphrase_that_copies_the_passage_terms(self):
        copied = question("a", "paraphrase", gold(10, 10, anchor=("urea",)), text="urea retained blood sharks water")
        honest = question("b", "paraphrase", gold(10, 10, anchor=("urea",)), text="why do cartilaginous fish stay hydrated")
        self.assertError("leakage", copied)
        self.assertEqual(self.errors(honest)[0], [])

    def test_low_term_overlap_for_a_lexical_question_is_only_a_warning(self):
        errors, warnings = self.errors(question("a", "lexical", gold(10, 10, anchor=("urea",)), text="osmosis puzzle"))
        self.assertEqual(errors, [])
        self.assertTrue(any("lexical question shares" in w for w in warnings))

    def test_coverage_problems_are_warnings(self):
        errors, warnings = self.errors(question("a", "lexical", gold(10, 10, anchor=("urea",)), text="urea"))
        self.assertEqual(errors, [])
        self.assertTrue(any("second.pdf has 0 answerable" in w for w in warnings))


class RankingOverrides(unittest.TestCase):
    def test_dotted_keys_neutral_and_order(self):
        config = ev.parse_ranking(["low_quality_weight=0.8", "type_weights.figure=0.9"])
        self.assertEqual((config.low_quality_weight, config.type_weights), (0.8, {"figure": 0.9}))
        self.assertEqual(ev.parse_ranking(["low_quality_weight=0.8", "neutral"]), NEUTRAL)
        self.assertEqual(ev.parse_ranking(["neutral", "table_degraded_weight=0.5"]).table_degraded_weight, 0.5)
        self.assertEqual(ev.parse_ranking(None), RankingConfig())

    def test_unknown_keys_bad_values_and_negative_weights_are_rejected(self):
        for item in ("nonsense=1", "type_weights.video=1", "low_quality_weight=abc", "low_quality_weight=-1",
                     "type_weights.figure=-0.5", "justtext"):
            with self.subTest(item), self.assertRaises(ValueError):
                ev.parse_ranking([item])

    def test_the_saved_result_records_the_full_ranking_config(self):
        passages = [passage(1, "alpha beta"), passage(2, "alpha gamma")]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.jsonl"
            path.write_text("", encoding="utf-8")
            result = ev.evaluate(Index(passages), [question("q1", "lexical", gold(1, 1, anchor=("beta",)))],
                                 passages, path, "dev", "anchor", root=Path(directory),
                                 ranking=RankingConfig(low_quality_weight=0.7))
        self.assertEqual(result["ranking"], {"low_quality_weight": 0.7, "table_degraded_weight": 1.0,
                                             "credit_only_weight": 1.0, "type_weights": {}})

    def test_a_sweep_reports_one_row_per_value_and_how_many_were_tried(self):
        passages = [passage(1, "alpha beta", low_quality=True), passage(2, "alpha beta")]
        questions = [question("q1", "lexical", gold(2, 2, anchor=("beta",)))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "q.jsonl"
            path.write_text("", encoding="utf-8")
            index = Index(passages)
            reference = ev.evaluate(index, questions, passages, path, "dev", "anchor", root=Path(directory),
                                    ranking=NEUTRAL)
            rows = ev.sweep(index, questions, passages, path, "dev", "anchor", RankingConfig(),
                            "low_quality_weight", ["1.0", "0.9", "0.5"], reference)
        self.assertEqual(len(rows), 1 + 1 + 3 + 1)
        self.assertIn("3 configurations tried", rows[-1])


class Cli(unittest.TestCase):
    def test_holdout_needs_final(self):
        with self.assertRaises(SystemExit), patch("sys.stderr"):
            ev.main(["--split", "holdout"])

    def test_the_cli_and_the_evaluator_build_the_corpus_through_the_same_function(self):
        self.assertIs(cli.load_corpus, load_corpus)
        self.assertIs(ev.load_corpus, load_corpus)

    def test_load_corpus_reads_the_given_directory_only(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "n.txt").write_text("coral reef", encoding="utf-8")
            self.assertEqual(load_corpus(Path(directory)), load_passages(Path(directory)))


class SelfTest(unittest.TestCase):
    def test_a_synthetic_corpus_retrieves_itself(self):
        passages = [passage(i, " ".join(f"{w}{i}" for w in ("alpha", "beta", "gamma", "delta", "kappa", "omega",
                                                              "sigma", "theta")) + ". " + " ".join(
            f"filler{i}x{j}" for j in range(40)), index=i) for i in range(1, 21)]
        result = ev.selftest(passages, sample=20, seed=1, min_words=40)
        self.assertEqual((result["n"], result["share"]), (20, 1.0))
        self.assertEqual(result["per_book"][SRC]["share"], 1.0)


class SearchScored(unittest.TestCase):
    def test_search_returns_the_same_passages_in_the_same_order_as_search_scored(self):
        passages = [passage(i, "coral reef " * (i % 3 + 1) + "fish " * i, index=i) for i in range(1, 12)]
        index = Index(passages)
        scored = index.search_scored("coral fish", 6)
        self.assertEqual([p for p, _ in scored], index.search("coral fish", 6))
        self.assertEqual([p for p, _ in scored], retrieve("coral fish", passages, 6))
        self.assertTrue(all(a[1] >= b[1] for a, b in zip(scored, scored[1:])))


if __name__ == "__main__":
    unittest.main()
