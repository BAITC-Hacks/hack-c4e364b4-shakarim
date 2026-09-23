"""End-to-end integration tests for the merged financial network pipeline."""
from __future__ import annotations

import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import networkx as nx
import numpy as np
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


METRIC_COLUMNS = {
    "gid", "depth", "is_seed", "in_degree", "out_degree", "in_amount", "out_amount",
    "in_tx_count", "out_tx_count", "unique_senders", "unique_receivers", "pass_through",
    "pagerank", "betweenness", "truncated_by_depth",
}


def sample_tables():
    """Reciprocal transfers, a censored leaf, and a seed with no transfers."""
    a, b, c, isolated, e = range(9223372036854775701, 9223372036854775706)
    nodes = pd.DataFrame({
        "gid": pd.Series([a, b, c, isolated, e], dtype="int64"),
        "depth": [0, 1, 4, 0, 4],
        "is_seed": [True, False, False, True, False],
    })
    edges = pd.DataFrame({
        "src": pd.Series([a, b, b, b, e], dtype="int64"),
        "dst": pd.Series([b, a, c, e, b], dtype="int64"),
        "sum_kzt": [100.0, 20.0, 60.0, 10.0, 5.0],
        "n_tx": [2, 1, 2, 1, 1],
        "depth": [1, 1, 4, 4, 4],
    })
    tx = pd.DataFrame({
        "src": pd.Series([a, a, b, b, b, b, e], dtype="int64"),
        "dst": pd.Series([b, b, a, c, c, e, b], dtype="int64"),
        "date": pd.date_range("2025-01-01", periods=7),
        "sum_kzt": [40.0, 60.0, 20.0, 25.0, 35.0, 10.0, 5.0],
    })
    return edges, nodes, tx


class LoaderValidationTests(unittest.TestCase):
    def assert_invalid(self, tables):
        with self.assertRaises((ValueError, TypeError)):
            loader.sanity_check(*tables)

    def test_valid_fixture_reports_isolated_gid(self):
        edges, nodes, tx = sample_tables()
        self.assertEqual(loader.sanity_check(edges, nodes, tx), {int(nodes.gid.iloc[3])})

    def test_required_columns_cannot_be_missing_or_null(self):
        required = (
            (0, ("src", "dst", "sum_kzt", "n_tx", "depth")),
            (1, ("gid", "depth", "is_seed")),
            (2, ("src", "dst", "date", "sum_kzt")),
        )
        for table_index, columns in required:
            for column in columns:
                with self.subTest(table=table_index, column=column, error="missing"):
                    tables = list(sample_tables())
                    tables[table_index] = tables[table_index].drop(columns=column)
                    self.assert_invalid(tables)
                with self.subTest(table=table_index, column=column, error="null"):
                    tables = list(sample_tables())
                    tables[table_index][column] = tables[table_index][column].astype(object)
                    tables[table_index].at[0, column] = None
                    self.assert_invalid(tables)

    def test_duplicate_nodes_and_ordered_edges_are_rejected(self):
        for table_index in (0, 1):
            with self.subTest(table=table_index):
                tables = list(sample_tables())
                tables[table_index] = pd.concat(
                    [tables[table_index], tables[table_index].iloc[[0]]], ignore_index=True,
                )
                self.assert_invalid(tables)

    def test_unknown_transaction_and_edge_endpoints_are_rejected(self):
        for table_index in (0, 2):
            for endpoint in ("src", "dst"):
                with self.subTest(table=table_index, endpoint=endpoint):
                    tables = list(sample_tables())
                    tables[table_index].at[0, endpoint] = 123
                    self.assert_invalid(tables)

    def test_ids_cannot_be_float_string_bool_or_unsigned_overflow(self):
        for table_index, column in ((1, "gid"), (0, "src"), (0, "dst"), (2, "src"), (2, "dst")):
            for dtype in ("float64", "str", "bool"):
                with self.subTest(table=table_index, column=column, dtype=dtype):
                    tables = list(sample_tables())
                    tables[table_index][column] = tables[table_index][column].astype(dtype)
                    self.assert_invalid(tables)
            with self.subTest(table=table_index, column=column, error="int64 overflow"):
                tables = list(sample_tables())
                tables[table_index][column] = tables[table_index][column].astype("uint64")
                tables[table_index].at[0, column] = np.uint64(2**63)
                self.assert_invalid(tables)

    def test_amounts_are_positive_and_finite(self):
        for table_index in (0, 2):
            for value in (0.0, -1.0, np.nan, np.inf, -np.inf):
                with self.subTest(table=table_index, amount=value):
                    tables = list(sample_tables())
                    tables[table_index].at[0, "sum_kzt"] = value
                    self.assert_invalid(tables)

    def test_counts_and_depths_cannot_be_fractional_or_out_of_range(self):
        for table_index, column, invalid_values in (
            (0, "n_tx", (0, -1, 1.5, np.inf)),
            (0, "depth", (-1, 1.5, np.inf)),
            (1, "depth", (-1, 1.5, np.inf)),
        ):
            for value in invalid_values:
                with self.subTest(table=table_index, column=column, value=value):
                    tables = list(sample_tables())
                    tables[table_index][column] = tables[table_index][column].astype("float64")
                    tables[table_index].at[0, column] = value
                    self.assert_invalid(tables)

    def test_seed_flag_requires_booleans(self):
        for value in ("False", "True", 1, 0):
            with self.subTest(seed=value):
                edges, nodes, tx = sample_tables()
                nodes["is_seed"] = nodes["is_seed"].astype(object)
                nodes.at[0, "is_seed"] = value
                self.assert_invalid((edges, nodes, tx))

    def test_invalid_dates_are_rejected(self):
        edges, nodes, tx = sample_tables()
        tx["date"] = tx["date"].astype(object)
        tx.at[0, "date"] = "not-a-date"
        self.assert_invalid((edges, nodes, tx))

    def test_aggregates_match_each_directed_pair_not_only_global_totals(self):
        edges, nodes, tx = sample_tables()
        edges.at[0, "sum_kzt"] += 1
        edges.at[1, "sum_kzt"] -= 1
        self.assertAlmostEqual(edges.sum_kzt.sum(), tx.sum_kzt.sum())
        self.assert_invalid((edges, nodes, tx))

        edges, nodes, tx = sample_tables()
        edges.at[0, "n_tx"] -= 1
        edges.at[1, "n_tx"] += 1
        self.assertEqual(edges.n_tx.sum(), len(tx))
        self.assert_invalid((edges, nodes, tx))

    def test_missing_transaction_pairs_are_rejected(self):
        edges, nodes, tx = sample_tables()
        tx.at[0, "dst"] = nodes.gid.iloc[2]
        self.assert_invalid((edges, nodes, tx))

    def test_explicit_missing_directory_does_not_fall_back_to_repo_data(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing-data"
            with self.assertRaises(FileNotFoundError):
                loader.load(missing)

    def test_load_validates_actual_parquet_and_preserves_exact_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            edges, nodes, tx = sample_tables()
            for frame, name in zip((edges, nodes, tx), ("edges", "nodes", "transactions")):
                frame.to_parquet(directory / f"{name}.parquet", index=False)
            loaded_edges, loaded_nodes, loaded_tx = loader.load(directory)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(loaded_tx.date))
            for actual, expected, columns in (
                (loaded_edges, edges, ("src", "dst")),
                (loaded_nodes, nodes, ("gid",)),
                (loaded_tx, tx, ("src", "dst")),
            ):
                for column in columns:
                    self.assertEqual(actual[column].dtype, np.dtype("int64"))
                    self.assertListEqual(actual[column].tolist(), expected[column].tolist())
            edges.at[0, "sum_kzt"] += 1
            edges.to_parquet(directory / "edges.parquet", index=False)
            with self.assertRaises((ValueError, TypeError)):
                loader.load(directory)

    def test_safe_integer_id_dtypes_normalize_to_int64(self):
        for dtype in ("int32", "uint64"):
            with self.subTest(dtype=dtype), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                edges, nodes, tx = sample_tables()
                mapping = {gid: index + 1 for index, gid in enumerate(nodes.gid)}
                for frame, columns in ((edges, ("src", "dst")), (nodes, ("gid",)), (tx, ("src", "dst"))):
                    for column in columns:
                        frame[column] = frame[column].map(mapping).astype(dtype)
                for frame, name in zip((edges, nodes, tx), ("edges", "nodes", "transactions")):
                    frame.to_parquet(directory / f"{name}.parquet", index=False)
                loaded = loader.load(directory)
                for frame, columns in zip(loaded, (("src", "dst"), ("gid",), ("src", "dst"))):
                    for column in columns:
                        self.assertEqual(frame[column].dtype, np.dtype("int64"))


class GraphAndMetricsTests(unittest.TestCase):
    def test_directed_graph_preserves_isolates_and_attributes(self):
        edges, nodes, _ = sample_tables()
        graph = graph_builder.build_graph(edges, nodes)
        self.assertIsInstance(graph, nx.DiGraph)
        self.assertEqual(set(graph), set(nodes.gid))
        self.assertEqual(graph.number_of_edges(), 5)
        self.assertEqual(set(nx.isolates(graph)), {int(nodes.gid.iloc[3])})
        for row in nodes.itertuples(index=False):
            self.assertEqual(graph.nodes[row.gid]["depth"], row.depth)
            self.assertEqual(graph.nodes[row.gid]["is_seed"], row.is_seed)
        for row in edges.itertuples(index=False):
            self.assertEqual(graph[row.src][row.dst]["sum_kzt"], row.sum_kzt)
            self.assertEqual(graph[row.src][row.dst]["n_tx"], row.n_tx)
            self.assertEqual(graph[row.src][row.dst]["depth"], row.depth)

    def test_graph_builder_refuses_silent_pair_overwrite_or_unknown_node(self):
        edges, nodes, _ = sample_tables()
        duplicate_edges = pd.concat([edges, edges.iloc[[0]]], ignore_index=True)
        with self.assertRaises((ValueError, TypeError)):
            graph_builder.build_graph(duplicate_edges, nodes)
        edges.at[0, "src"] = 123
        with self.assertRaises((ValueError, TypeError)):
            graph_builder.build_graph(edges, nodes)

    def test_reciprocal_flows_and_boundary_flags(self):
        edges, nodes, _ = sample_tables()
        graph = graph_builder.build_graph(edges, nodes)
        result = metrics.compute_node_metrics(graph, nodes, edges).set_index("gid")
        a, b, c, isolated, e = nodes.gid.tolist()
        expected = {
            a: (1, 1, 20.0, 100.0, 1, 2),
            b: (2, 3, 105.0, 90.0, 3, 4),
            c: (1, 0, 60.0, 0.0, 2, 0),
            isolated: (0, 0, 0.0, 0.0, 0, 0),
            e: (1, 1, 10.0, 5.0, 1, 1),
        }
        columns = ["in_degree", "out_degree", "in_amount", "out_amount", "in_tx_count", "out_tx_count"]
        for gid, values in expected.items():
            with self.subTest(gid=gid):
                self.assertListEqual(result.loc[gid, columns].tolist(), list(values))
                self.assertEqual(result.at[gid, "unique_senders"], values[0])
                self.assertEqual(result.at[gid, "unique_receivers"], values[1])
        self.assertAlmostEqual(result.at[b, "pass_through"], 90 / 105)
        self.assertAlmostEqual(result.at[e, "pass_through"], 0.5)
        self.assertEqual(result.at[c, "pass_through"], 0)
        self.assertTrue(pd.isna(result.at[a, "pass_through"]))
        self.assertTrue(pd.isna(result.at[isolated, "pass_through"]))
        self.assertEqual(set(result.index[result.truncated_by_depth]), {c})
        self.assertEqual(result.out_amount.sum(), 195.0)
        self.assertEqual(result.in_amount.sum(), 195.0)

    def test_zero_inflow_nonseed_has_no_pass_through_ratio(self):
        edges, nodes, _ = sample_tables()
        nodes.at[3, "is_seed"] = False
        graph = graph_builder.build_graph(edges, nodes)
        result = metrics.compute_node_metrics(graph, nodes, edges).set_index("gid")
        self.assertTrue(pd.isna(result.at[int(nodes.gid.iloc[3]), "pass_through"]))

    def test_centrality_uses_amount_weights_and_inverse_amount_distances(self):
        edges, nodes, _ = sample_tables()
        graph = graph_builder.build_graph(edges, nodes)
        with patch.object(metrics.nx, "pagerank", wraps=nx.pagerank) as pagerank_call, patch.object(
            metrics.nx, "betweenness_centrality", wraps=nx.betweenness_centrality,
        ) as betweenness_call:
            result = metrics.compute_node_metrics(graph, nodes, edges)
        self.assertEqual(pagerank_call.call_args.kwargs["weight"], "sum_kzt")
        weight = betweenness_call.call_args.kwargs["weight"]
        weighted_graph = betweenness_call.call_args.args[0]
        for _, _, attributes in weighted_graph.edges(data=True):
            self.assertAlmostEqual(attributes[weight], 1 / attributes["sum_kzt"])
        self.assertAlmostEqual(result.pagerank.sum(), 1.0)
        self.assertTrue(result.pagerank.between(0, 1).all())
        self.assertTrue(result.betweenness.between(0, 1).all())

    def test_weighted_betweenness_prefers_larger_transfers(self):
        # A->B->C has distance 0.02; the small direct A->C transfer costs 1.0.
        nodes = pd.DataFrame({"gid": [1, 2, 3], "depth": [0, 1, 2], "is_seed": [True, False, False]})
        edges = pd.DataFrame({
            "src": [1, 2, 1], "dst": [2, 3, 3], "sum_kzt": [100.0, 100.0, 1.0],
            "n_tx": [1, 1, 1], "depth": [1, 2, 2],
        })
        graph = graph_builder.build_graph(edges, nodes)
        result = metrics.compute_node_metrics(graph, nodes, edges).set_index("gid")
        self.assertAlmostEqual(result.at[2, "betweenness"], 0.5)
        self.assertEqual(result.at[1, "betweenness"], 0)
        self.assertEqual(result.at[3, "betweenness"], 0)

    def test_csv_roundtrip_keeps_adjacent_int64_gids_exact(self):
        edges, nodes, _ = sample_tables()
        result = metrics.compute_node_metrics(graph_builder.build_graph(edges, nodes), nodes, edges)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "node_metrics.csv"
            metrics.save_node_metrics(result, path)
            restored = pd.read_csv(path)
        self.assertEqual(restored.gid.dtype, np.dtype("int64"))
        self.assertListEqual(restored.gid.tolist(), nodes.gid.tolist())
        self.assertTrue(METRIC_COLUMNS.issubset(restored.columns))
        self.assertTrue(restored.loc[restored.is_seed, "pass_through"].isna().all())


class PipelineIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = loader.find_data_dir()
        cls._temporary_output = tempfile.TemporaryDirectory(prefix="dev1-integration-")
        cls.addClassCleanup(cls._temporary_output.cleanup)
        cls.output_dir = Path(cls._temporary_output.name)
        cls.edges, cls.nodes, cls.tx = loader.load(cls.data_dir)
        started = time.perf_counter()
        pipeline.run_from_parquet(cls.data_dir, cls.output_dir)
        cls.pipeline_seconds = time.perf_counter() - started

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

    def test_real_dataset_counts_and_complete_gid_coverage(self):
        self.assertEqual(len(self.nodes), 2248)
        self.assertEqual(len(self.edges), 3119)
        self.assertEqual(len(self.tx), 4840)
        self.assertEqual(len(self.metrics_df), 2248)
        self.assertEqual(set(self.metrics_df.gid), set(self.nodes.gid))
        self.assertEqual(self.metrics_df.gid.dtype, np.dtype("int64"))

    def test_real_dataset_flow_totals_and_per_pair_reconciliation(self):
        expected_total = float(self.edges.sum_kzt.sum())
        for column in ("in_amount", "out_amount"):
            np.testing.assert_allclose(self.metrics_df[column].sum(), expected_total, rtol=1e-12)
        for column in ("in_tx_count", "out_tx_count"):
            self.assertEqual(self.metrics_df[column].sum(), 4840)
        actual = self.tx.groupby(["src", "dst"]).agg(sum_kzt=("sum_kzt", "sum"), n_tx=("sum_kzt", "size"))
        expected = self.edges.set_index(["src", "dst"])[["sum_kzt", "n_tx"]].sort_index()
        actual = actual.sort_index()
        pd.testing.assert_index_equal(actual.index, expected.index)
        np.testing.assert_allclose(actual.sum_kzt, expected.sum_kzt, rtol=1e-12, atol=0.01)
        np.testing.assert_array_equal(actual.n_tx, expected.n_tx)

    def test_real_seed_and_boundary_nodes_are_not_misinterpreted(self):
        result = self.metrics_df
        self.assertTrue(result.loc[result.is_seed, "pass_through"].isna().all())
        isolated = result[(result.in_degree == 0) & (result.out_degree == 0)]
        self.assertEqual(len(isolated), 19)
        self.assertTrue(isolated.is_seed.all())
        self.assertEqual(isolated[["in_amount", "out_amount", "in_tx_count", "out_tx_count"]].to_numpy().sum(), 0)
        boundary = (result.depth >= 4) & (result.out_degree == 0)
        self.assertEqual(int(boundary.sum()), 444)
        pd.testing.assert_series_equal(result.truncated_by_depth, boundary, check_names=False)
        observed = result[~result.is_seed & (result.in_amount > 0)]
        np.testing.assert_allclose(observed.pass_through, observed.out_amount / observed.in_amount)

    def test_complete_pipeline_finishes_within_five_minutes(self):
        self.assertLess(self.pipeline_seconds, 300, f"Pipeline took {self.pipeline_seconds:.2f}s")

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
