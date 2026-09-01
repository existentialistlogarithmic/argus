"""Normalization and similarity — the layer every identity decision rests on."""

import unittest
from datetime import date

from argus.normalize import (
    NICKNAMES, norm_address, norm_date, norm_email, norm_id, norm_name,
    norm_org, norm_phone, soundex,
)
from argus.similarity import (
    date_similarity, edit_ratio, jaro_winkler, levenshtein, name_similarity,
    org_similarity, token_set_ratio,
)


class TestNormalizeNames(unittest.TestCase):
    def test_expands_nicknames(self):
        self.assertEqual(norm_name("Bob Whitlock"), "robert whitlock")
        self.assertEqual(norm_name("Liz Ashford"), "elizabeth ashford")

    def test_strips_honorifics_and_suffixes(self):
        self.assertEqual(norm_name("Mr. James Brennan Jr."), "james brennan")
        self.assertEqual(norm_name("Dr Katherine Novak PhD"), "katherine novak")

    def test_folds_accents_and_case(self):
        self.assertEqual(norm_name("JOSÉ MARCHETTI"), "jose marchetti")

    def test_preserves_token_order(self):
        # Order-insensitivity belongs to the comparator, not the normalizer.
        self.assertEqual(norm_name("Smith, John"), "smith john")

    def test_nickname_table_maps_to_canonical_forms(self):
        for nickname, canonical in NICKNAMES.items():
            self.assertEqual(norm_name(nickname), canonical, nickname)


class TestNormalizeOrgs(unittest.TestCase):
    def test_strips_legal_forms_only(self):
        self.assertEqual(norm_org("Meridian Holdings Ltd"), "meridian holdings")
        self.assertEqual(norm_org("Meridian Group Inc"), "meridian group")

    def test_distinctive_words_survive(self):
        # These two are unrelated firms; over-normalizing would fuse them.
        self.assertNotEqual(norm_org("Kestrel Holdings Ltd"), norm_org("Kestrel Trading Ltd"))

    def test_punctuated_legal_form_collapses(self):
        self.assertEqual(norm_org("Cobalt Ventures S.A."), "cobalt ventures")
        self.assertEqual(norm_org("Cobalt Ventures SA"), "cobalt ventures")


class TestNormalizeIdentifiers(unittest.TestCase):
    def test_phone_reduces_to_last_ten_digits(self):
        for rendering in ("+44 712 345 6789", "0712-345-6789", "(712) 3456789", "7123456789"):
            self.assertEqual(norm_phone(rendering), "7123456789", rendering)

    def test_email_folds_gmail_dots_and_plus_tags(self):
        self.assertEqual(norm_email("John.Smith+work@Gmail.com"), "johnsmith@gmail.com")
        self.assertEqual(norm_email("john.smith@outlook.com"), "john.smith@outlook.com")

    def test_id_strips_punctuation(self):
        self.assertEqual(norm_id("GB-1234 5678"), "gb12345678")

    def test_date_parses_several_formats(self):
        for rendering in ("1974-03-08", "08/03/1974", "08 Mar 1974", "19740308"):
            self.assertEqual(norm_date(rendering), date(1974, 3, 8), rendering)

    def test_date_returns_none_on_garbage(self):
        self.assertIsNone(norm_date("not a date"))
        self.assertIsNone(norm_date(""))

    def test_address_expands_abbreviations(self):
        self.assertEqual(norm_address("12 Seething Ln"), norm_address("12 Seething Lane"))

    def test_soundex_groups_homophones(self):
        self.assertEqual(soundex("Whitlock"), soundex("Whitlok"))
        self.assertNotEqual(soundex("Whitlock"), soundex("Brennan"))


class TestStringSimilarity(unittest.TestCase):
    def test_levenshtein(self):
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(levenshtein("same", "same"), 0)

    def test_edit_ratio_bounds(self):
        self.assertEqual(edit_ratio("abc", "abc"), 1.0)
        self.assertEqual(edit_ratio("abc", "xyz"), 0.0)

    def test_jaro_winkler_rewards_shared_prefix(self):
        self.assertGreater(jaro_winkler("martha", "marhta"), 0.95)
        self.assertLess(jaro_winkler("abcde", "fghij"), 0.1)

    def test_token_set_ratio_is_order_insensitive(self):
        self.assertEqual(token_set_ratio("john smith", "smith john"), 1.0)


class TestNameSimilarity(unittest.TestCase):
    def test_missing_middle_name_still_matches(self):
        # The regression that motivated best-first token alignment: walking
        # tokens in order let the middle initial consume the surname slot.
        score = name_similarity(norm_name("Robert J Whitlock"), norm_name("Bob Whitlock"))
        self.assertGreater(score, 0.88)

    def test_initial_matches_full_given_name(self):
        self.assertGreater(name_similarity("j smith", "john smith"), 0.88)

    def test_reversed_order_matches(self):
        self.assertGreater(name_similarity("smith john", "john smith"), 0.88)

    def test_different_given_name_does_not_reach_agreement(self):
        self.assertLess(name_similarity("john smith", "jane smith"), 0.88)

    def test_typo_still_matches(self):
        self.assertGreater(name_similarity("michael underwood", "micheal underwood"), 0.88)

    def test_empty_is_zero(self):
        self.assertEqual(name_similarity("", "john smith"), 0.0)


class TestOrgSimilarity(unittest.TestCase):
    def test_shared_prefix_is_not_a_match(self):
        # Company names share leading words constantly; a prefix-boosted
        # character comparator fuses unrelated firms here.
        # Below legal_name's disagree_below (0.85), so these score as full
        # disagreement rather than earning partial credit toward a merge.
        self.assertLess(org_similarity("ardent partners", "ardent ventures"), 0.85)
        self.assertLess(org_similarity("meridian holdings", "meridian trading"), 0.85)

    def test_typo_still_matches(self):
        self.assertGreater(org_similarity("ardent enterpriess", "ardent enterprises"), 0.88)

    def test_identical_is_one(self):
        self.assertEqual(org_similarity("kestrel consulting", "kestrel consulting"), 1.0)


class TestDateSimilarity(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(date_similarity(date(1980, 5, 4), date(1980, 5, 4)), 1.0)

    def test_day_month_transposition_is_near_exact(self):
        self.assertGreater(date_similarity(date(1980, 5, 4), date(1980, 4, 5)), 0.9)

    def test_off_by_one_day_keeps_credit(self):
        self.assertGreaterEqual(date_similarity(date(1980, 5, 4), date(1980, 5, 5)), 0.9)

    def test_different_years_score_zero(self):
        self.assertEqual(date_similarity(date(1980, 5, 4), date(1991, 11, 2)), 0.0)

    def test_non_dates_score_zero(self):
        self.assertEqual(date_similarity(None, date(1980, 5, 4)), 0.0)


if __name__ == "__main__":
    unittest.main()
