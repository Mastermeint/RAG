import unittest
from collections import Counter

from rag_mvp.textclean import (build_vocabulary, clean_chars, is_noise, join_fragments, join_lines, repair_ocr,
                               text_quality)


class CharacterRepair(unittest.TestCase):
    def test_ligatures_hyphen_variants_spaces_and_invisible_characters(self):
        self.assertEqual(clean_chars("\ufb01sh \ufb02at e\ufb00ort o\ufb03ce"), "fish flat effort office")
        self.assertEqual(clean_chars("muskel\u2011 lunge"), "muskel- lunge")
        self.assertEqual(clean_chars("a\xa0b\u2003c\u200bd\x00e"), "a b cde")


class LineJoining(unittest.TestCase):
    def test_discretionary_soft_hyphen_rejoins_the_word(self):
        self.assertEqual(join_lines(["the sys\xad", "tem works"]), "the system works")

    def test_explicit_hyphen_plus_soft_hyphen_is_a_real_compound(self):
        self.assertEqual(join_lines(["a long-\xad", "lost sub"]), "a long-lost sub")

    def test_plain_hyphen_break_is_joined_by_default(self):
        self.assertEqual(join_lines(["the sys-", "tem works"]), "the system works")

    def test_vocabulary_keeps_genuine_compounds(self):
        vocab = build_vocabulary(["it is a well-known fact", "the well-known result"])
        self.assertEqual(join_lines(["a well-", "known fact"], vocab), "a well-known fact")

    def test_vocabulary_does_not_block_ordinary_hyphenation(self):
        vocab = build_vocabulary(["the system is large", "a well-known fact"])
        self.assertEqual(join_lines(["the sys-", "tem is"], vocab), "the system is")

    def test_hyphen_before_a_capital_is_kept(self):
        self.assertEqual(join_fragments("the Indo-", "Pacific region"), "the Indo-Pacific region")

    def test_ordinary_lines_join_with_one_space(self):
        self.assertEqual(join_lines(["alpha  ", "  beta"]), "alpha beta")


class OcrRepair(unittest.TestCase):
    VOCAB = Counter({"the": 500, "these": 60, "may": 120, "that": 200, "spawn": 30, "mill": 40, "hole": 25})

    def test_systematic_confusions_are_repaired_with_case_preserved(self):
        self.assertEqual(repair_ocr("TI1e rnay tllat spaw11 TI1ese", self.VOCAB), "The may that spawn These")

    def test_known_rare_and_numeric_tokens_are_left_alone(self):
        text = "mill Lagenorhynchus H2O 1,000 2011 hole"
        self.assertEqual(repair_ocr(text, self.VOCAB), text)

    def test_needs_strong_evidence_from_the_vocabulary(self):
        # 'tllat' -> 'that' only because 'that' is frequent; a candidate seen twice is not enough
        self.assertEqual(repair_ocr("tllat", Counter({"that": 2})), "tllat")


class Quality(unittest.TestCase):
    CLEAN = "The eggs remain in the sand until the next high tide when they hatch and are washed out to sea."

    def test_clean_prose_is_not_poor(self):
        self.assertFalse(text_quality(self.CLEAN)["poor"])

    def test_ocr_damage_is_poor(self):
        damaged = "TI1e eggs remain in the sand until spaw11 Han1lets tllat T11e ~ | ^ usual {shape} of a reef fr0nt"
        self.assertTrue(text_quality(damaged)["poor"])

    def test_symbol_soup_is_poor(self):
        self.assertTrue(text_quality("~ ~| ^ { } _ < > \\ | ~ ~ ^^ || {{ }} __ << >> ~ ^ |" * 2)["poor"])

    def test_noise_filter_drops_speckle_but_keeps_short_real_text(self):
        self.assertTrue(is_noise(",;  a;  ~ ! i'l"))
        self.assertTrue(is_noise("12 34 56"))
        self.assertFalse(is_noise("Coral reefs"))


if __name__ == "__main__":
    unittest.main()
