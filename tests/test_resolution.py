"""Ontology loading, Fellegi-Sunter scoring, and conflict-aware clustering."""

import math
import unittest

from argus.model import Record, merge_props
from argus.ontology import BlockingRule, Ontology, PropertyDef, load_default
from argus.resolve import Resolver, Weights, evaluate

MINIMAL = {
    "name": "test",
    "entities": {
        "person": {
            "match_threshold": 9.0,
            "review_threshold": 4.5,
            "properties": {
                "full_name": {"kind": "name", "m_prob": 0.92, "u_prob": 0.03, "threshold": 0.88},
                "dob": {"kind": "date", "m_prob": 0.88, "u_prob": 0.0008, "threshold": 0.90},
                "national_id": {"kind": "id", "m_prob": 0.98, "u_prob": 1e-6, "threshold": 1.0},
                "email": {"kind": "email", "m_prob": 0.75, "u_prob": 1e-4, "threshold": 1.0,
                          "multi": True},
            },
            "blocking": [
                {"name": "name_phon", "properties": ["full_name"], "strategy": "soundex"},
                {"name": "natid", "properties": ["national_id"], "strategy": "exact"},
            ],
        },
        "account": {"properties": {"account_number": {"kind": "id", "u_prob": 1e-7}}},
    },
    "links": {
        "owns": {"source": "person", "target": "account"},
    },
}


def make_ontology():
    return Ontology.from_dict(MINIMAL)


def person(key, **raw):
    ontology = make_ontology()
    entity_type = ontology.entity("person")
    source, local = key.split(":", 1)
    return Record(source=source, local_id=local, entity_type="person",
                  raw=raw, props=entity_type.normalize_props(raw))


class TestOntology(unittest.TestCase):
    def test_default_ontology_loads_and_validates(self):
        ontology = load_default()
        self.assertIn("person", ontology.entities)
        self.assertIn("transacted", ontology.links)
        self.assertIn("ontology", ontology.summary())

    def test_rejects_link_to_unknown_entity(self):
        document = {"entities": {"a": {}}, "links": {"l": {"source": "a", "target": "ghost"}}}
        with self.assertRaises(ValueError):
            Ontology.from_dict(document)

    def test_rejects_blocking_on_unknown_property(self):
        document = {
            "entities": {"a": {"properties": {"x": {}},
                               "blocking": [{"name": "b", "properties": ["nope"]}]}},
            "links": {},
        }
        with self.assertRaises(ValueError):
            Ontology.from_dict(document)

    def test_rejects_review_threshold_above_match(self):
        document = {"entities": {"a": {"match_threshold": 1.0, "review_threshold": 5.0}}, "links": {}}
        with self.assertRaises(ValueError):
            Ontology.from_dict(document)

    def test_unknown_type_lookup_raises(self):
        with self.assertRaises(KeyError):
            make_ontology().entity("dragon")

    def test_partial_floor_defaults_below_threshold(self):
        prop = PropertyDef(name="x", threshold=0.9)
        self.assertAlmostEqual(prop.partial_floor, 0.81)
        self.assertEqual(PropertyDef(name="x", threshold=0.9, disagree_below=0.5).partial_floor, 0.5)

    def test_multi_valued_similarity_takes_best_pair(self):
        prop = PropertyDef(name="email", kind="email", multi=True)
        score = prop.similarity(["a@x.com", "b@x.com"], ["b@x.com"])
        self.assertEqual(score, 1.0)


class TestBlocking(unittest.TestCase):
    def test_exact_rule_produces_one_key(self):
        rule = BlockingRule(name="natid", properties=("national_id",))
        self.assertEqual(rule.keys({"national_id": "ab123"}), ["natid:ab123"])

    def test_missing_value_produces_no_key(self):
        rule = BlockingRule(name="natid", properties=("national_id",))
        self.assertEqual(rule.keys({}), [])

    def test_soundex_rule_groups_similar_names(self):
        rule = BlockingRule(name="p", properties=("full_name",), strategy="soundex")
        self.assertEqual(rule.keys({"full_name": "robert whitlock"}),
                         rule.keys({"full_name": "robert whitlok"}))


class TestWeights(unittest.TestCase):
    def setUp(self):
        self.weights = Weights(make_ontology().entity("person"))

    def test_agreement_weight_is_log_likelihood_ratio(self):
        self.assertAlmostEqual(self.weights.agree["full_name"], math.log2(0.92 / 0.03), places=6)

    def test_discriminating_property_outweighs_a_name(self):
        self.assertGreater(self.weights.agree["national_id"], self.weights.agree["full_name"] * 3)

    def test_disagreement_weight_is_negative(self):
        self.assertLess(self.weights.disagree["dob"], 0)

    def test_exclusive_set_holds_only_discriminating_single_valued_properties(self):
        self.assertIn("national_id", self.weights.exclusive)
        self.assertNotIn("full_name", self.weights.exclusive)
        self.assertNotIn("email", self.weights.exclusive)  # multi-valued

    def test_evidence_interpolates_between_floor_and_threshold(self):
        agree = self.weights.evidence("full_name", 1.0)
        disagree = self.weights.evidence("full_name", 0.0)
        partial = self.weights.evidence("full_name", 0.86)
        self.assertEqual(agree, self.weights.agree["full_name"])
        self.assertEqual(disagree, self.weights.disagree["full_name"])
        self.assertTrue(disagree < partial < agree)


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.resolver = Resolver(make_ontology())

    def test_name_alone_is_not_enough_to_merge(self):
        # A name-only agreement lands in the review band: worth an analyst's
        # look, never an automatic merge.
        decision = self.resolver.score(
            person("a:1", full_name="Robert Whitlock"),
            person("b:1", full_name="Robert Whitlock"),
        )
        self.assertEqual(decision.verdict, "review")
        self.assertLess(decision.score, 9.0)

    def test_name_plus_dob_matches(self):
        decision = self.resolver.score(
            person("a:1", full_name="Robert Whitlock", dob="1974-03-08"),
            person("b:1", full_name="Bob Whitlock", dob="08/03/1974"),
        )
        self.assertEqual(decision.verdict, "match")

    def test_shared_name_with_different_dob_is_rejected(self):
        decision = self.resolver.score(
            person("a:1", full_name="Robert Whitlock", dob="1974-03-08"),
            person("b:1", full_name="Robert Whitlock", dob="1991-11-02"),
        )
        self.assertEqual(decision.verdict, "no-match")

    def test_missing_data_is_not_disagreement(self):
        both = self.resolver.score(
            person("a:1", full_name="Robert Whitlock", dob="1974-03-08"),
            person("b:1", full_name="Robert Whitlock", dob="1974-03-08"),
        )
        one_missing = self.resolver.score(
            person("a:1", full_name="Robert Whitlock", dob="1974-03-08"),
            person("b:1", full_name="Robert Whitlock"),
        )
        self.assertGreater(both.score, one_missing.score)
        self.assertGreater(one_missing.score, 0)

    def test_national_id_alone_is_conclusive(self):
        decision = self.resolver.score(
            person("a:1", national_id="K37186034"),
            person("b:1", national_id="K37186034"),
        )
        self.assertEqual(decision.verdict, "match")

    def test_decision_explains_itself(self):
        decision = self.resolver.score(
            person("a:1", full_name="Robert Whitlock", dob="1974-03-08"),
            person("b:1", full_name="Bob Whitlock", dob="1974-03-08"),
        )
        text = decision.explain()
        self.assertIn("full_name", text)
        self.assertIn("dob", text)
        self.assertEqual(len(decision.evidence), 2)


class TestClustering(unittest.TestCase):
    def setUp(self):
        self.resolver = Resolver(make_ontology())

    def resolve(self, records):
        return self.resolver.resolve({"person": records})

    def test_merges_a_true_cluster(self):
        records = [
            person("natreg:1", full_name="Robert J Whitlock", dob="1974-03-08"),
            person("bank:1", full_name="Bob Whitlock", dob="1974-03-08"),
            person("hr:1", full_name="Robert Whitlock", dob="08/03/1974"),
        ]
        result = self.resolve(records)
        self.assertEqual(len(result.entities), 1)

    def test_conflicting_exclusive_identifier_vetoes_the_merge(self):
        # A and B match on name+dob, B and C match on name+dob, but A and C
        # hold different passports. Plain transitive closure fuses all three.
        records = [
            person("a:1", full_name="Robert Whitlock", dob="1974-03-08", national_id="AAA111"),
            person("b:1", full_name="Robert Whitlock", dob="1974-03-08"),
            person("c:1", full_name="Robert Whitlock", dob="1974-03-08", national_id="BBB222"),
        ]
        result = self.resolve(records)
        self.assertGreater(len(result.entities), 1)
        self.assertGreater(result.stats["by_type"]["person"]["conflicts_blocked"], 0)

    def test_distinct_people_stay_distinct(self):
        records = [
            person("a:1", full_name="Robert Whitlock", dob="1974-03-08"),
            person("b:1", full_name="Robert Whitlock", dob="1991-11-02"),
        ]
        self.assertEqual(len(self.resolve(records).entities), 2)

    def test_every_record_is_assigned_exactly_once(self):
        records = [
            person("a:1", full_name="Robert Whitlock", dob="1974-03-08"),
            person("b:1", full_name="Katherine Novak", dob="1980-01-01"),
            person("c:1", full_name="Bob Whitlock", dob="1974-03-08"),
        ]
        result = self.resolve(records)
        self.assertEqual(set(result.assignment), {r.key for r in records})
        members = [m for e in result.entities.values() for m in e.members]
        self.assertEqual(len(members), len(set(members)))

    def test_evaluation_reports_perfect_scores_on_a_clean_split(self):
        records = [
            person("a:1", full_name="Robert Whitlock", dob="1974-03-08"),
            person("b:1", full_name="Bob Whitlock", dob="1974-03-08"),
            person("c:1", full_name="Katherine Novak", dob="1980-01-01"),
        ]
        records[0].truth_id = records[1].truth_id = "T1"
        records[2].truth_id = "T2"
        report = evaluate({"person": records}, self.resolve(records))
        self.assertEqual(report["overall"]["precision"], 1.0)
        self.assertEqual(report["overall"]["recall"], 1.0)


class TestMergeProps(unittest.TestCase):
    def test_single_valued_takes_the_most_common(self):
        entity_type = make_ontology().entity("person")
        records = [
            person("a:1", full_name="Robert Whitlock"),
            person("b:1", full_name="Robert Whitlock"),
            person("c:1", full_name="Bobby Whitlock"),
        ]
        self.assertEqual(merge_props(entity_type, records)["full_name"], "robert whitlock")

    def test_multi_valued_takes_the_union(self):
        entity_type = make_ontology().entity("person")
        records = [
            person("a:1", email="a@x.com"),
            person("b:1", email="b@x.com"),
            person("c:1", email="a@x.com"),
        ]
        self.assertEqual(sorted(merge_props(entity_type, records)["email"]),
                         ["a@x.com", "b@x.com"])


if __name__ == "__main__":
    unittest.main()
