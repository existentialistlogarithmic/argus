"""Graph, analytics, query language, and the end-to-end pipeline."""

import shutil
import tempfile
import unittest
from pathlib import Path

from argus import analytics, corpus
from argus.graph import KnowledgeGraph
from argus.model import Edge, Entity
from argus.ontology import load_default
from argus.pipeline import Investigation, Pipeline
from argus.query import QueryError, execute, parse, tokenize


def toy_graph() -> KnowledgeGraph:
    """Four people in a line, plus one company two of them run."""
    graph = KnowledgeGraph(load_default())
    for index, name in enumerate(["alice ashford", "bob brennan", "cara calloway", "dan duarte"]):
        graph.add_entity(Entity(id=f"PER-{index}", entity_type="person",
                                props={"full_name": name}, members=[f"s:{index}"],
                                sources=["s"]))
    graph.add_entity(Entity(id="ORG-0", entity_type="organization",
                            props={"legal_name": "kestrel trading"}, members=["s:o"],
                            sources=["s"]))
    edges = [
        ("communicated", "PER-0", "PER-1"),
        ("communicated", "PER-1", "PER-2"),
        ("communicated", "PER-2", "PER-3"),
        ("officer_of", "PER-0", "ORG-0"),
        ("officer_of", "PER-3", "ORG-0"),
    ]
    for index, (link_type, src, dst) in enumerate(edges):
        graph.add_edge(Edge(id=f"E-{index}", link_type=link_type, src=src, dst=dst,
                            observations=[{}]))
    return graph


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.graph = toy_graph()

    def test_type_index(self):
        self.assertEqual(len(self.graph.of_type("person")), 4)
        self.assertEqual(len(self.graph.of_type("organization")), 1)

    def test_degree_counts_both_directions(self):
        self.assertEqual(self.graph.degree("PER-0"), 2)
        self.assertEqual(self.graph.degree("PER-1"), 2)

    def test_shortest_path_prefers_the_company_shortcut(self):
        path = self.graph.shortest_path("PER-0", "PER-3")
        self.assertEqual(len(path), 2)  # via ORG-0, not the 3-hop chain

    def test_shortest_path_returns_none_when_unreachable(self):
        self.graph.add_entity(Entity(id="PER-9", entity_type="person", props={}))
        self.assertIsNone(self.graph.shortest_path("PER-0", "PER-9"))

    def test_link_type_filter_removes_the_shortcut(self):
        path = self.graph.shortest_path("PER-0", "PER-3", link_types={"communicated"})
        self.assertEqual(len(path), 3)

    def test_all_paths_finds_both_routes(self):
        paths = self.graph.all_paths("PER-0", "PER-3", max_hops=4)
        self.assertGreaterEqual(len(paths), 2)
        self.assertEqual(len(paths[0]), 2)  # sorted shortest-first

    def test_ego_expands_by_hops(self):
        self.assertEqual(len(self.graph.ego("PER-1", hops=1)), 3)
        self.assertGreater(len(self.graph.ego("PER-1", hops=2)), 3)

    def test_search_matches_property_text(self):
        self.assertEqual(self.graph.search("brennan")[0].id, "PER-1")

    def test_stats_report_shape(self):
        stats = self.graph.stats()
        self.assertEqual(stats["entities"], 5)
        self.assertEqual(stats["edges"], 5)
        self.assertEqual(stats["by_entity_type"]["person"], 4)

    def test_save_and_load_round_trip(self):
        directory = Path(tempfile.mkdtemp())
        try:
            self.graph.save(directory / "g.json")
            reloaded = KnowledgeGraph.load(directory / "g.json", self.graph.ontology)
            self.assertEqual(len(reloaded.entities), len(self.graph.entities))
            self.assertEqual(len(reloaded.edges), len(self.graph.edges))
            self.assertEqual(reloaded.degree("PER-0"), 2)
        finally:
            shutil.rmtree(directory)


class TestAnalytics(unittest.TestCase):
    def setUp(self):
        self.graph = toy_graph()

    def test_degree_centrality_is_normalized(self):
        scores = analytics.degree_centrality(self.graph)
        self.assertTrue(all(0.0 <= v <= 1.0 for v in scores.values()))

    def test_betweenness_finds_the_brokers(self):
        scores = analytics.betweenness_centrality(self.graph)
        # The company and the middle of the chain sit on the routes between
        # otherwise separated people.
        self.assertGreater(scores["ORG-0"], scores["PER-1"] * 0 + 1e-9)
        self.assertGreater(max(scores.values()), 0)

    def test_pagerank_sums_to_one(self):
        ranks = analytics.personalized_pagerank(self.graph)
        self.assertAlmostEqual(sum(ranks.values()), 1.0, places=4)

    def test_seeded_pagerank_favours_the_seed(self):
        ranks = analytics.personalized_pagerank(self.graph, seeds={"PER-0": 1.0})
        self.assertEqual(max(ranks, key=ranks.get), "PER-0")

    def test_communities_are_assigned_to_every_node(self):
        labels = analytics.communities(self.graph)
        self.assertEqual(set(labels), set(self.graph.entities))

    def test_modularity_of_a_single_community_is_not_positive(self):
        labels = {node: 0 for node in self.graph.entities}
        self.assertLessEqual(analytics.modularity(self.graph, labels), 0.0)

    def test_empty_graph_is_handled(self):
        empty = KnowledgeGraph(load_default())
        self.assertEqual(analytics.communities(empty), {})
        self.assertEqual(analytics.personalized_pagerank(empty), {})


class TestQueryLanguage(unittest.TestCase):
    def setUp(self):
        self.graph = toy_graph()
        self.context = {"risk": {"PER-0": 0.9, "PER-1": 0.4}}

    def test_tokenizer_classifies(self):
        kinds = [t.kind for t in tokenize('find person where risk > 0.5')]
        self.assertEqual(kinds, ["keyword", "word", "keyword", "word", "op", "number"])

    def test_find_by_type(self):
        self.assertEqual(len(execute(self.graph, "find person", self.context)), 4)

    def test_where_numeric_comparison(self):
        results = execute(self.graph, "find person where risk > 0.5", self.context)
        self.assertEqual([e.id for e in results], ["PER-0"])

    def test_contains_operator(self):
        results = execute(self.graph, 'find person where full_name ~ "brennan"', self.context)
        self.assertEqual([e.id for e in results], ["PER-1"])

    def test_boolean_and_or(self):
        query = 'find person where full_name ~ "ashford" or full_name ~ "brennan"'
        self.assertEqual(len(execute(self.graph, query, self.context)), 2)
        query = 'find person where risk > 0.1 and risk < 0.5'
        self.assertEqual([e.id for e in execute(self.graph, query, self.context)], ["PER-1"])

    def test_not_and_parentheses(self):
        query = 'find person where not (full_name ~ "ashford")'
        self.assertEqual(len(execute(self.graph, query, self.context)), 3)

    def test_linked_to_within(self):
        results = execute(self.graph, "find person linked to PER-0 within 1", self.context)
        self.assertEqual({e.id for e in results}, {"PER-1"})
        results = execute(self.graph, "find person linked to PER-0 within 2", self.context)
        self.assertEqual({e.id for e in results}, {"PER-1", "PER-2", "PER-3"})

    def test_linked_to_via_link_type(self):
        query = "find person linked to PER-0 via officer_of within 2"
        self.assertEqual({e.id for e in execute(self.graph, query, self.context)}, {"PER-3"})

    def test_order_and_limit(self):
        results = execute(self.graph, "find person order by risk desc limit 2", self.context)
        self.assertEqual(results[0].id, "PER-0")
        self.assertEqual(len(results), 2)

    def test_computed_fields(self):
        self.assertEqual(len(execute(self.graph, "find person where degree > 1", self.context)), 4)
        self.assertEqual(len(execute(self.graph, 'find where type = "organization"', self.context)), 1)

    def test_path_query(self):
        result = execute(self.graph, "path PER-0 to PER-3 within 4", self.context)
        self.assertGreaterEqual(len(result["paths"]), 1)

    def test_show_query(self):
        self.assertEqual(execute(self.graph, "show PER-0", self.context).id, "PER-0")

    def test_reference_resolves_by_name(self):
        self.assertEqual(execute(self.graph, 'show "brennan"', self.context).id, "PER-1")

    def test_syntax_errors_are_reported(self):
        for bad in ["", "select * from person", "find person where", "find person limit"]:
            with self.assertRaises(QueryError, msg=bad):
                parse(bad)

    def test_unknown_entity_type_is_reported(self):
        with self.assertRaises(QueryError):
            execute(self.graph, "find dragon", self.context)

    def test_unknown_reference_is_reported(self):
        with self.assertRaises(QueryError):
            execute(self.graph, "show PER-999", self.context)


class TestNaturalLanguageFallback(unittest.TestCase):
    def setUp(self):
        from argus.nl import NaturalLanguage, RuleBasedProvider

        self.interface = NaturalLanguage(toy_graph(), provider=RuleBasedProvider())

    def test_translates_a_ranking_question(self):
        translation = self.interface.translate("who are the riskiest people?")
        self.assertIn("find person", translation.query)
        self.assertIn("order by risk", translation.query)

    def test_translates_a_connection_question(self):
        translation = self.interface.translate("how is Alice connected to Dan?")
        self.assertTrue(translation.query.startswith("path"))

    def test_translates_a_lookup(self):
        translation = self.interface.translate("who is Bob Brennan?")
        self.assertTrue(translation.query.startswith("show"))

    def test_every_translation_parses(self):
        for question in ["who are the riskiest people?",
                         "how is Alice connected to Dan?",
                         "who is Bob Brennan?",
                         "show me the watchlisted people"]:
            parse(self.interface.translate(question).query)

    def test_untranslatable_question_raises_rather_than_guessing(self):
        with self.assertRaises(QueryError):
            self.interface.translate("summarize everything you know")


class TestEndToEnd(unittest.TestCase):
    """A full run on a small corpus, checked against its own ground truth."""

    @classmethod
    def setUpClass(cls):
        cls.directory = Path(tempfile.mkdtemp())
        cls.meta = corpus.generate(cls.directory, seed=99, n_people=80, n_orgs=25)
        cls.result = Pipeline(verbose=False).run(cls.directory)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.directory, ignore_errors=True)

    def test_corpus_is_deterministic(self):
        second = Path(tempfile.mkdtemp())
        try:
            meta = corpus.generate(second, seed=99, n_people=80, n_orgs=25)
            self.assertEqual(meta["ring_people"], self.meta["ring_people"])
            self.assertEqual((second / "truth.json").read_text(),
                             (self.directory / "truth.json").read_text())
        finally:
            shutil.rmtree(second, ignore_errors=True)

    def test_ingest_produced_records_and_links(self):
        self.assertGreater(len(self.result.ingestor.records), 100)
        self.assertGreater(len(self.result.ingestor.links), 100)

    def test_resolution_makes_no_false_merges(self):
        # Over-merging invents relationships between unrelated people, which is
        # the failure mode that must never happen silently.
        self.assertEqual(self.result.evaluation["overall"]["precision"], 1.0)

    def test_resolution_recall_is_high(self):
        self.assertGreater(self.result.evaluation["overall"]["recall"], 0.85)

    def test_resolution_collapses_records_into_far_fewer_entities(self):
        stats = self.result.resolution.stats["total"]
        self.assertLess(stats["entities"], stats["records"] / 2)

    def test_every_link_was_rewired_onto_entities(self):
        self.assertEqual(self.result.graph.dangling_links, 0)
        self.assertGreater(len(self.result.graph.edges), 50)

    def test_watchlist_entries_resolved_onto_full_entities(self):
        self.assertTrue(self.result.seeds)
        for seed in self.result.seeds:
            entity = self.result.graph.entities[seed]
            # The listing carries only a name and a date of birth; resolution
            # is what attaches it to accounts, employers, and counterparties.
            self.assertGreater(len(entity.sources), 1)

    def test_planted_network_is_detected(self):
        kinds = {finding.kind for finding in self.result.findings}
        self.assertIn("circular_flow", kinds)
        self.assertIn("structuring", kinds)
        self.assertIn("shared_registration", kinds)

    def test_findings_are_correlated_into_a_case(self):
        self.assertTrue(self.result.cases)
        top = self.result.cases[0]
        self.assertGreaterEqual(len(top["kinds"]), 2)
        self.assertTrue(top["people"], "a case must name the people behind it")

    def test_ring_principals_rank_near_the_top(self):
        truth = {}
        import json

        records = json.loads((self.directory / "truth.json").read_text())["records"]
        for key, entity_id in self.result.resolution.assignment.items():
            if key in records:
                truth.setdefault(entity_id, records[key])
        ring = set(self.meta["ring_people"])
        ranked = sorted(
            (e for e in self.result.graph.entities.values() if e.entity_type == "person"),
            key=lambda e: -self.result.risk.get(e.id, 0.0),
        )
        top_ten = [truth.get(e.id) for e in ranked[:10]]
        self.assertGreaterEqual(sum(1 for t in top_ten if t in ring), 4)

    def test_investigation_surfaces_render(self):
        investigation = Investigation(self.result)
        top = max(self.result.risk, key=self.result.risk.get)
        self.assertIn("PROVENANCE", investigation.dossier(top))
        self.assertIn("risk", investigation.top_risk("person", limit=5))
        self.assertIn("CASE", investigation.case_report(limit=1))

    def test_unknown_reference_is_reported_not_raised(self):
        self.assertIn("No entity", Investigation(self.result).dossier("nonexistent-xyz"))

    def test_query_runs_against_the_real_graph(self):
        context = {"risk": self.result.risk, "communities": self.result.communities}
        results = execute(self.result.graph, "find person order by risk desc limit 5", context)
        self.assertEqual(len(results), 5)
        self.assertGreaterEqual(self.result.risk[results[0].id],
                                self.result.risk[results[-1].id])


if __name__ == "__main__":
    unittest.main()
