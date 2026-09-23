import re
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"
DATA = ROOT.parents[1] / "data (1)" / "data"
ROLES = {"consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"}


class OutputContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nodes = pd.read_parquet(DATA / "nodes.parquet")
        cls.edges = pd.read_parquet(DATA / "edges.parquet")
        cls.tx = pd.read_parquet(DATA / "transactions.parquet")
        cls.metrics = pd.read_csv(OUTPUT / "node_metrics.csv")
        cls.roles = pd.read_csv(OUTPUT / "nodes_roles.csv")
        cls.clusters = pd.read_csv(OUTPUT / "clusters.csv")
        cls.top = pd.read_csv(OUTPUT / "top_nodes.csv")

    def test_node_metrics_cover_every_gid_once(self):
        self.assertEqual(len(self.metrics), len(self.nodes))
        self.assertTrue(self.metrics.gid.is_unique)
        self.assertEqual(set(self.metrics.gid), set(self.nodes.gid))

    def test_input_aggregates_are_consistent(self):
        self.assertEqual(int(self.metrics.out_tx_count.sum()), len(self.tx))
        self.assertEqual(int(self.metrics.out_degree.sum()), len(self.edges))
        self.assertAlmostEqual(self.metrics.out_amount.sum(), self.edges.sum_kzt.sum(), places=2)

    def test_nodes_roles_contract(self):
        required = {"gid", "role", "role_score", "cluster_id", "priority_score", "evidence"}
        self.assertTrue(required.issubset(self.roles.columns))
        self.assertEqual(len(self.roles), 2248)
        self.assertFalse(self.roles[list(required)].isna().any().any())
        self.assertTrue(set(self.roles.role).issubset(ROLES))
        self.assertTrue(self.roles.role_score.between(0, 1).all())
        self.assertTrue(self.roles.priority_score.between(0, 1).all())
        self.assertTrue(self.roles.evidence.map(lambda value: bool(re.search(r"\d", value))).all())
        self.assertTrue(self.roles.evidence.str.len().le(200).all())

    def test_depth_boundary_is_not_called_terminal(self):
        boundary = self.roles[(self.roles.depth == 4) & (self.roles.out_deg == 0)]
        self.assertGreater(len(boundary), 0)
        self.assertTrue(boundary.truncated_by_depth.all())
        self.assertFalse(boundary.role.eq("terminal").any())

    def test_seed_pass_through_is_not_interpreted(self):
        seeds = self.metrics[self.metrics.is_seed]
        self.assertGreater(len(seeds), 0)
        self.assertTrue(seeds.pass_through.isna().all())

    def test_cluster_contract(self):
        required = {"cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"}
        self.assertTrue(required.issubset(self.clusters.columns))
        self.assertTrue(self.clusters.cluster_id.is_unique)
        self.assertEqual(self.clusters.n_nodes.sum(), len(self.nodes))
        self.assertFalse(self.clusters[list(required)].isna().any().any())

    def test_top_nodes_contract_and_sorting(self):
        required = {"rank", "gid", "role", "priority_score", "why"}
        self.assertTrue(required.issubset(self.top.columns))
        self.assertGreaterEqual(len(self.top), 20)
        self.assertListEqual(self.top["rank"].tolist(), list(range(1, len(self.top) + 1)))
        self.assertTrue(self.top.priority_score.is_monotonic_decreasing)


class StreamlitSmokeTest(unittest.TestCase):
    def test_app_starts_without_exception(self):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20).run()
        self.assertEqual(app.exception, [])
        self.assertEqual(len(app.tabs), 3)


if __name__ == "__main__":
    unittest.main()
