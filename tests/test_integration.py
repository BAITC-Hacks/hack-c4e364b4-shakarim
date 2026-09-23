"""End-to-end integration tests for the merged financial network pipeline."""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
import pandas as pd
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "ui"))

import loader
import graph_builder
import metrics
import roles
import clustering
import priority
import export
import pipeline


class PipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = loader.find_data_dir()
        cls.output_dir = ROOT / "output"
        cls.edges, cls.nodes, cls.tx = loader.load(cls.data_dir)
        pipeline.run_from_parquet(cls.data_dir, cls.output_dir)

        cls.metrics_df = pd.read_csv(cls.output_dir / "node_metrics.csv")
        cls.nodes_roles = pd.read_csv(cls.output_dir / "nodes_roles.csv")
        cls.clusters = pd.read_csv(cls.output_dir / "clusters.csv")
        cls.top_nodes = pd.read_csv(cls.output_dir / "top_nodes.csv")

    def test_data_loader(self):
        self.assertGreater(len(self.edges), 0)
        self.assertGreater(len(self.nodes), 0)
        self.assertGreater(len(self.tx), 0)
        orphans = loader.sanity_check(self.edges, self.nodes, self.tx)
        self.assertIsInstance(orphans, set)

    def test_graph_builder(self):
        G = graph_builder.build_graph(self.edges, self.nodes)
        self.assertEqual(G.number_of_nodes(), len(self.nodes))
        self.assertEqual(G.number_of_edges(), len(self.edges))

    def test_node_metrics_contract(self):
        required = {"gid", "depth", "is_seed", "in_degree", "out_degree", "in_amount", "out_amount",
                    "unique_senders", "unique_receivers", "in_tx_count", "out_tx_count",
                    "truncated_by_depth", "pass_through", "pagerank", "betweenness"}
        self.assertTrue(required.issubset(self.metrics_df.columns))
        self.assertEqual(len(self.metrics_df), len(self.nodes))
        self.assertTrue(self.metrics_df["gid"].is_unique)

    def test_nodes_roles_contract(self):
        expected_cols = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"]
        self.assertListEqual(list(self.nodes_roles.columns), expected_cols)
        self.assertEqual(len(self.nodes_roles), len(self.nodes))
        self.assertFalse(self.nodes_roles[expected_cols].isna().any().any())
        valid_roles = {"consolidator", "distributor", "transit", "terminal", "coordinator", "peripheral"}
        self.assertTrue(set(self.nodes_roles["role"]).issubset(valid_roles))
        self.assertTrue(self.nodes_roles["role_score"].between(0, 1).all())
        self.assertTrue(self.nodes_roles["priority_score"].between(0, 100).all())

    def test_clusters_contract(self):
        expected_cols = ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"]
        self.assertListEqual(list(self.clusters.columns), expected_cols)
        self.assertTrue(self.clusters["cluster_id"].is_unique)
        self.assertEqual(self.clusters["n_nodes"].sum(), len(self.nodes))
        self.assertFalse(self.clusters[expected_cols].isna().any().any())

    def test_top_nodes_contract(self):
        expected_cols = ["rank", "gid", "role", "priority_score", "why"]
        self.assertListEqual(list(self.top_nodes.columns), expected_cols)
        self.assertGreaterEqual(len(self.top_nodes), 20)
        self.assertListEqual(self.top_nodes["rank"].tolist(), list(range(1, len(self.top_nodes) + 1)))
        self.assertTrue(self.top_nodes["priority_score"].is_monotonic_decreasing)


class StreamlitUIIntegrationTests(unittest.TestCase):
    def test_app_renders_main_view(self):
        app = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=30).run()
        self.assertEqual(len(app.exception), 0)

    def test_search_and_select_client(self):
        app = AppTest.from_file(str(ROOT / "ui/app.py"), default_timeout=30).run()
        top_nodes = pd.read_csv(ROOT / "output/top_nodes.csv")
        test_gid = str(top_nodes.iloc[0]["gid"])
        app.sidebar.text_input[0].set_value(test_gid).run()
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
