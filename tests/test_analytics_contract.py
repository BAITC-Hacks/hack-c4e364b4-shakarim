"""Regression checks for censored inputs, community weights and report safety."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clustering import louvain, weighted_projection
from export import export
from pipeline import analyze, file_sha256, main, parse_edges
from priority import priority_components, priority_score
from roles import classify


def metrics(**updates):
    row = dict(gid="1", depth=2, is_seed=False, in_amount=10000, out_amount=9200,
               unique_senders=4, unique_receivers=4, in_tx_count=5, out_tx_count=5)
    return {**row, **updates}


class ObservationLimitsTests(unittest.TestCase):
    def test_seed_is_never_terminal_or_transit_for_incomplete_inflow(self):
        for incoming in (100, 10000, 1000000):
            with self.subTest(incoming=incoming):
                row = metrics(is_seed=True, in_amount=incoming)
                role, _, explanation = classify(row)
                self.assertNotIn(role, {"terminal", "transit"})
                self.assertIn("входящие неполны", explanation)
                self.assertEqual(priority_components(row)["imbalance"], 0)
                self.assertLessEqual(len(explanation), 200)

    def test_seed_structural_roles_do_not_depend_on_in_out_ratio(self):
        for senders, receivers, expected in ((8, 8, "coordinator"), (12, 3, "consolidator"), (3, 12, "distributor")):
            observed = []
            for amount in (1, 10000, 10000000):
                observed.append(classify(metrics(is_seed=True, in_amount=amount,
                    unique_senders=senders, unique_receivers=receivers))[0])
            self.assertEqual(observed, [expected] * 3)

    def test_censored_nodes_cannot_claim_retention_or_earn_imbalance_points(self):
        for senders in (1, 8, 24):
            row = metrics(depth=4, out_amount=0, unique_senders=senders, unique_receivers=0, out_tx_count=0)
            role, score, evidence = classify(row)
            self.assertEqual(role, "peripheral")
            self.assertLessEqual(score, .2)
            self.assertEqual(priority_components(row)["imbalance"], 0)
            self.assertIn("границы выгрузки", evidence)
            self.assertLessEqual(len(evidence), 200)

    def test_priority_is_finite_and_bounded_for_extreme_observed_inputs(self):
        for value in (0, 1, 1000000000000):
            row = metrics(in_amount=value, out_amount=0, unique_senders=value, unique_receivers=value)
            self.assertTrue(0 <= priority_score(row) <= 1)


class CommunityContractTests(unittest.TestCase):
    def test_projection_sums_reciprocal_and_parallel_weights_preserving_isolates(self):
        edges = [("1", "2", 10), ("2", "1", 20), ("1", "2", 3), ("1", "1", 7)]
        graph = weighted_projection(["1", "2", "3"], edges)
        self.assertFalse(graph.is_directed())
        self.assertEqual(graph["1"]["2"]["weight"], 33)
        self.assertEqual(graph.number_of_edges(), 1)
        self.assertEqual(graph.degree("3"), 0)
        self.assertEqual(louvain(["1", "2", "3"], edges), louvain(["3", "2", "1"], list(reversed(edges))))

    def test_directional_metrics_are_not_recomputed_from_clustering_edges(self):
        rows = [metrics(gid="1", in_amount=10000, out_amount=100), metrics(gid="2")]
        original = copy.deepcopy(rows)
        nodes, clusters = analyze(rows, [("1", "2", 5), ("2", "1", 6), ("1", "1", 7)])
        self.assertEqual(rows, original)
        for actual, row in zip(nodes, rows):
            self.assertEqual(actual["role"], classify(row)[0])
            self.assertEqual(actual["priority_score"], priority_score(row))
        self.assertEqual(sum(cluster["sum_kzt_internal"] for cluster in clusters), 18)


class InputAndExportTests(unittest.TestCase):
    def test_csv_aliases_are_exported_canonically_and_context_matches_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, edges, output = root / "input.csv", root / "edges.csv", root / "output"
            rows = [dict(node_id=f"1000000000000000{index:02}", seed=index == 0, node_depth=4,
                         incoming_amount=100, outgoing_amount=0, n_senders=3, n_receivers=0,
                         n_in=4, n_out=0) for index in range(20)]
            with inputs.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            edges.write_text(f"source_gid,target_gid,amount_kzt,n_tx\n{rows[0]['node_id']},{rows[1]['node_id']},100,4\n")
            self.assertEqual(main(["--input", str(inputs), "--edges", str(edges), "--output-dir", str(output)]), 0)
            with (output / "node_metrics.csv").open(encoding="utf-8-sig") as stream:
                restored = list(csv.DictReader(stream))
            self.assertEqual(restored[0]["gid"], rows[0]["node_id"])
            self.assertEqual(restored[0]["is_seed"], "True")
            self.assertEqual(restored[0]["depth"], "4")
            self.assertEqual(restored[0]["in_tx_count"], "4")
            with (output / "edges.csv").open(encoding="utf-8-sig") as stream:
                restored_edges = list(csv.DictReader(stream))
            self.assertEqual(restored_edges[0], {"src": rows[0]["node_id"], "dst": rows[1]["node_id"], "sum_kzt": "100.0", "n_tx": "4"})
            context = json.loads((output / "analysis_context.json").read_text())
            self.assertEqual(context["mode"], "metrics")
            self.assertIsNone(context["data_dir"])
            self.assertEqual(len(context["output_sha256"]), 5)
            for name, digest in context["output_sha256"].items():
                self.assertEqual(file_sha256(output / name), digest)
            # Running in the same directory is safe, and removes stale edges.
            self.assertEqual(main(["--input", str(output / "node_metrics.csv"), "--output-dir", str(output)]), 0)
            with (output / "edges.csv").open(encoding="utf-8-sig") as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])

    def test_invalid_weight_and_metric_are_rejected_instead_of_zeroed(self):
        for value in ("bad", "", "nan", "inf", "-1"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_edges([dict(src="1", dst="2", sum_kzt=value)], {"1", "2"})
                with self.assertRaises(ValueError):
                    analyze([metrics(in_amount=value)])
        with self.assertRaises(ValueError):
            analyze([metrics(is_seed="unknown")])

    def test_failed_export_does_not_overwrite_existing_reports(self):
        nodes, clusters = analyze([metrics(gid=str(index)) for index in range(20)])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            export(nodes, clusters, directory)
            before = {path.name: path.read_bytes() for path in directory.iterdir()}
            for field, value in (("priority_score", 3), ("role_score", float("nan")), ("evidence", "x" * 201)):
                broken = copy.deepcopy(nodes)
                broken[0][field] = value
                with self.assertRaises(ValueError):
                    export(broken, clusters, directory)
                self.assertEqual({path.name: path.read_bytes() for path in directory.iterdir()}, before)

    def test_top_explanation_contains_actual_priority_contributions(self):
        nodes, _ = analyze([metrics()])
        for component in ("оборот", "связи", "дисбаланс", "seed"):
            self.assertIn(component, nodes[0]["why"])


if __name__ == "__main__":
    unittest.main()
