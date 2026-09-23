"""Regression tests for the analytics owned by DEV 2.

Run with: python3.12 -m unittest discover -s tests -v
Only the Python standard library is required.
"""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clustering import louvain
from export import CLUSTERS_COLUMNS, NODES_ROLES_COLUMNS, TOP_NODES_COLUMNS
from pipeline import analyze, main, parse_edges
from priority import priority_score
from roles import MAX_EVIDENCE_LENGTH, ROLE_RULES, classify


class RoleClassificationTests(unittest.TestCase):
    def test_role_rules_are_documented_in_evaluation_order(self):
        self.assertEqual(len(ROLE_RULES), 7)
        self.assertTrue(ROLE_RULES[0].startswith("peripheral"))
        self.assertTrue(ROLE_RULES[1].startswith("terminal"))
        self.assertTrue(ROLE_RULES[2].startswith("coordinator"))

    def test_assigns_each_of_six_roles_for_explainable_examples(self):
        examples = {
            "consolidator": {"in_amount": 10_000, "out_amount": 3_000, "unique_senders": 11, "unique_receivers": 2},
            "distributor": {"in_amount": 500, "out_amount": 10_000, "unique_senders": 2, "unique_receivers": 11},
            "transit": {"in_amount": 10_000, "out_amount": 9_200, "unique_senders": 4, "unique_receivers": 4},
            "terminal": {"in_amount": 10_000, "out_amount": 100, "unique_senders": 5, "unique_receivers": 0, "depth": 2, "out_degree": 0},
            "coordinator": {"in_amount": 10_000, "out_amount": 10_000, "unique_senders": 10, "unique_receivers": 10},
            "peripheral": {"in_amount": 0, "out_amount": 0, "unique_senders": 0, "unique_receivers": 1},
        }
        for expected, metrics in examples.items():
            with self.subTest(role=expected):
                role, score, evidence = classify(metrics)
                self.assertEqual(role, expected)
                self.assertGreaterEqual(score, 0.05)
                self.assertLessEqual(score, 1.0)
                self.assertTrue(evidence)
                self.assertLessEqual(len(evidence), MAX_EVIDENCE_LENGTH)

    def test_evidence_has_concrete_values_and_is_at_most_two_hundred_characters(self):
        _, _, evidence = classify({
            "in_amount": 10_000, "out_amount": 9_200, "unique_senders": 4,
            "unique_receivers": 4, "in_tx_count": 4, "out_tx_count": 4,
        })
        self.assertIn("10,000", evidence)
        self.assertIn("92%", evidence)
        self.assertLessEqual(len(evidence), 200)

    def test_terminal_precedes_consolidator_for_low_pass_through(self):
        role, _, evidence = classify({
            "in_amount": 10_000, "out_amount": 1_000, "unique_senders": 10,
            "unique_receivers": 1, "in_tx_count": 10, "out_tx_count": 1, "depth": 2,
        })
        self.assertEqual(role, "terminal")
        self.assertIn("10%", evidence)

    def test_coordinator_precedes_transit_when_both_rules_match(self):
        role, _, _ = classify({
            "in_amount": 10_000, "out_amount": 9_200, "unique_senders": 6,
            "unique_receivers": 6, "in_tx_count": 6, "out_tx_count": 6,
        })
        self.assertEqual(role, "coordinator")

    def test_depth_four_zero_out_degree_is_not_terminal_evidence(self):
        metrics = {
            "depth": 4, "in_degree": 8, "out_degree": 0,
            "in_amount": 1000, "out_amount": 0,
            "unique_senders": 8, "unique_receivers": 0,
        }
        role, _, evidence = classify(metrics)
        self.assertNotEqual(role, "terminal")
        self.assertIn("границы выгрузки", evidence)


class PriorityTests(unittest.TestCase):
    def test_seed_adds_one_tenth_and_score_is_bounded(self):
        metrics = {"in_amount": 500_000, "out_amount": 250_000, "unique_senders": 4, "unique_receivers": 5}
        without_seed = priority_score(metrics)
        with_seed = priority_score({**metrics, "is_seed": "true"})
        self.assertAlmostEqual(with_seed - without_seed, .1, places=3)
        self.assertGreaterEqual(with_seed, 0)
        self.assertLessEqual(with_seed, 1)


class ClusteringTests(unittest.TestCase):
    def test_louvain_keeps_disconnected_transaction_groups_separate(self):
        nodes = ["a", "b", "c", "x", "y", "z"]
        edges = [("a", "b", 5), ("b", "c", 5), ("a", "c", 5),
                 ("x", "y", 5), ("y", "z", 5), ("x", "z", 5)]
        clusters = louvain(nodes, edges)
        self.assertEqual(clusters["a"], clusters["b"])
        self.assertEqual(clusters["b"], clusters["c"])
        self.assertEqual(clusters["x"], clusters["y"])
        self.assertEqual(clusters["y"], clusters["z"])
        self.assertNotEqual(clusters["a"], clusters["x"])

    def test_isolated_nodes_get_deterministic_cluster_ids(self):
        first = louvain(["z", "a", "b"], [])
        second = louvain(["b", "z", "a"], [])
        self.assertEqual(first, second)
        self.assertEqual(len(set(first.values())), 3)


class PipelineTests(unittest.TestCase):
    def test_requires_one_precomputed_metrics_row_per_gid(self):
        with self.assertRaisesRegex(ValueError, "должна содержать gid"):
            analyze([{"source_gid": "a", "target_gid": "b", "amount_kzt": 10}])
        with self.assertRaisesRegex(ValueError, "повторяется gid"):
            analyze([{"gid": "a"}, {"gid": "a"}])

    def test_uses_weighted_undirected_projection_when_edges_are_supplied(self):
        metrics = [{"gid": gid, "in_amount": 100, "out_amount": 90,
                    "unique_senders": 3, "unique_receivers": 3}
                   for gid in ("a", "b", "c", "x", "y", "z")]
        edges = [("a", "b", 10), ("b", "c", 10), ("c", "a", 10),
                 ("x", "y", 10), ("y", "z", 10), ("z", "x", 10)]
        nodes, clusters = analyze(metrics, edge_rows=edges)
        cluster_by_gid = {row["gid"]: row["cluster_id"] for row in nodes}
        self.assertEqual(cluster_by_gid["a"], cluster_by_gid["b"])
        self.assertNotEqual(cluster_by_gid["a"], cluster_by_gid["x"])
        self.assertEqual(sum(cluster["sum_kzt_internal"] for cluster in clusters), 60)

    def test_rejects_edges_outside_metrics_scope(self):
        with self.assertRaisesRegex(ValueError, "вне node_metrics.csv"):
            parse_edges([{"source": "known", "target": "unknown", "amount": 1}], {"known"})

    def test_exports_required_files_and_top_twenty_from_node_metrics(self):
        fields = ["gid", "depth", "is_seed", "in_degree", "out_degree", "in_amount", "out_amount",
                  "unique_senders", "unique_receivers", "in_tx_count", "out_tx_count", "pass_ratio",
                  "pagerank", "betweenness"]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path, output_dir = temp / "node_metrics.csv", temp / "output"
            with input_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index in range(25):
                    writer.writerow({"gid": f"gid-{index:02}", "depth": index % 5,
                                     "is_seed": index == 0, "in_degree": index % 7,
                                     "out_degree": (index + 1) % 6, "in_amount": 1000 + index * 100,
                                     "out_amount": 500 + index * 100, "unique_senders": index % 7,
                                     "unique_receivers": (index + 1) % 6, "in_tx_count": index % 7,
                                     "out_tx_count": (index + 1) % 6, "pass_ratio": 0.5,
                                     "pagerank": 0.01, "betweenness": 0})

            result = main(["--input", str(input_path), "--output-dir", str(output_dir)])
            self.assertEqual(result, 0)
            expected_headers = {
                "nodes_roles.csv": NODES_ROLES_COLUMNS,
                "clusters.csv": CLUSTERS_COLUMNS,
                "top_nodes.csv": TOP_NODES_COLUMNS,
            }
            for filename, headers in expected_headers.items():
                with self.subTest(file=filename):
                    with (output_dir / filename).open(encoding="utf-8-sig", newline="") as stream:
                        reader = csv.DictReader(stream)
                        rows = list(reader)
                    self.assertEqual(tuple(reader.fieldnames), headers)
                    self.assertEqual(len(rows), 25)
            with (output_dir / "top_nodes.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual([row["rank"] for row in csv.DictReader(stream)], [str(i) for i in range(1, 26)])

    def test_pipeline_rejects_fewer_than_twenty_nodes(self):
        fields = ["gid", "in_amount", "out_amount"]
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "small.csv"
            with input_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for index in range(19):
                    writer.writerow({"gid": index, "in_amount": 10, "out_amount": 5})
            self.assertEqual(main(["--input", str(input_path), "--output-dir", str(temp / "out")]), 2)
            self.assertFalse((temp / "out" / "top_nodes.csv").exists())


if __name__ == "__main__":
    unittest.main()
