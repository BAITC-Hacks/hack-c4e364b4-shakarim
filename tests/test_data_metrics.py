"""Additional regression coverage for the public metrics CSV contract."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.graph_builder import build_graph
from src.metrics import compute_node_metrics, save_node_metrics


class PassRatioContractTests(unittest.TestCase):
    def test_contract_alias_survives_csv_with_censored_seed_ratios(self):
        # The seed has observed inflow, but omitted incoming transfers still make
        # its outflow/inflow ratio unsuitable for role inference.
        nodes = pd.DataFrame({
            "gid": [1, 2, 3, 4],
            "depth": [0, 1, 4, 0],
            "is_seed": [True, False, False, True],
        })
        edges = pd.DataFrame({
            "src": [1, 2, 2], "dst": [2, 1, 3],
            "sum_kzt": [100.0, 10.0, 70.0],
            "n_tx": [1, 1, 1], "depth": [1, 1, 4],
        })
        result = compute_node_metrics(build_graph(edges, nodes), nodes, edges)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "node_metrics.csv"
            save_node_metrics(result, path)
            exported = pd.read_csv(path).set_index("gid")

        self.assertEqual(set(exported.index), {1, 2, 3, 4})
        pd.testing.assert_series_equal(
            exported.pass_ratio, exported.pass_through, check_names=False,
        )
        self.assertTrue(pd.isna(exported.at[1, "pass_ratio"]))
        self.assertTrue(pd.isna(exported.at[4, "pass_ratio"]))
        self.assertAlmostEqual(exported.at[2, "pass_ratio"], 0.8)
        self.assertEqual(exported.at[3, "pass_ratio"], 0.0)
        self.assertTrue(exported.at[3, "truncated_by_depth"])


if __name__ == "__main__":
    unittest.main()
