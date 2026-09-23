"""Regression checks for the judge's search and evidence workflow."""
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui"))

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest
import app
from components import investigation_brief, normalise_id
from graph_view import _network_canvas, load_edges


class ContractTests(unittest.TestCase):
    def test_invalid_edge_export_does_not_fall_back_to_another_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            primary, fallback = Path(directory) / "edges.csv", Path(directory) / "fallback.csv"
            fallback.write_text("src,dst,sum_kzt\n1,2,5000\n", encoding="utf-8")
            for body in ("wrong,columns\n1,2\n", "src,dst,sum_kzt\n1,2,inf\n", "src,dst,sum_kzt\n1,2,-1\n"):
                with self.subTest(body=body):
                    primary.write_text(body, encoding="utf-8")
                    st.cache_data.clear()
                    edges, path = load_edges((primary, fallback))
                    self.assertIsNone(path)
                    self.assertTrue(edges.empty)
                    self.assertIn("load_error", edges.attrs)

    def test_missing_edges_and_valid_empty_edges_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edges.csv"
            missing, _ = load_edges((path,))
            self.assertIn("load_error", missing.attrs)
            path.write_text("src,dst,sum_kzt\n", encoding="utf-8")
            st.cache_data.clear()
            empty, loaded_path = load_edges((path,))
            self.assertEqual(loaded_path, path)
            self.assertNotIn("load_error", empty.attrs)

    def test_brief_preserves_exact_gid_evidence_and_all_edges_with_caveats(self):
        gid = "9223372036854775807"
        client = pd.Series({"gid": gid, "role": "transit", "role_score": .83, "priority_score": .42,
                            "cluster_id": "07", "evidence": "Оригинальная гипотеза команды.", "depth": 4, "is_seed": True})
        edges = pd.DataFrame({"src": [gid] * 12, "dst": [str(i) for i in range(12)], "sum_kzt": [5000] * 12})
        brief = investigation_brief(client, "Оригинальный приоритет.", edges, "Демонстрационный кейс")
        self.assertIn("GID: " + gid, brief)
        self.assertIn("Всего направленных связей: 12", brief)
        self.assertIn(gid + " → 11", brief)
        for text in ("0.42 / 1", "Кластер: 07", "Оригинальная гипотеза команды.", "Оригинальный приоритет.", "SEED", "DEPTH=4", "Демонстрационный кейс"):
            self.assertIn(text, brief)

    def test_adjacent_large_int64_ids_remain_distinct_text_in_parquet_loader(self):
        ids = [9007199254740992, 9007199254740993]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "edges.parquet"
            pd.DataFrame({"src": ids, "dst": ids[::-1], "sum_kzt": [5000., 6000.]}).to_parquet(path)
            edges, _ = load_edges((path,))
            self.assertEqual(edges.src.tolist(), [str(gid) for gid in ids])
            self.assertEqual(edges.dst.tolist(), [str(gid) for gid in ids[::-1]])
        nodes = app.prepare_nodes(pd.DataFrame({"gid": ids}))
        self.assertEqual(nodes.gid.tolist(), [str(gid) for gid in ids])

    def test_int64_identifier_is_not_rounded(self):
        self.assertEqual(normalise_id("9223372036854775807.0"), "9223372036854775807")

    def test_invalid_scores_and_empty_evidence_are_rejected(self):
        nodes, _, _, _ = app.demo_results()
        nodes.loc[0, "role_score"] = 1.1
        nodes.loc[1, "evidence"] = " "
        issues = app.validate_nodes(nodes)
        self.assertTrue(any("role_score" in item for item in issues))
        self.assertTrue(any("evidence" in item for item in issues))

    def test_graph_escapes_external_ids_and_keeps_directions(self):
        nodes, _, _, edges = app.demo_results()
        nodes.loc[0, "gid"] = "901245"
        html = _network_canvas("<script>alert(1)</script>", edges.head(1), nodes, False)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("marker-end", html)
        self.assertIn("zoom-reset", html)

    def test_mock_contract_and_graph_integrity(self):
        nodes, clusters, top, edges = app.demo_results()
        self.assertFalse(app.validate_nodes(nodes))
        self.assertFalse(app.validate_optional(clusters, app.CLUSTER_COLUMNS, "clusters.csv"))
        self.assertFalse(app.validate_optional(top, app.TOP_NODE_COLUMNS, "top_nodes.csv"))
        self.assertGreaterEqual(len(top), 20)
        self.assertTrue(top.priority_score.is_monotonic_decreasing)
        self.assertTrue(set(top.gid).issubset(set(nodes.gid)))
        self.assertTrue((set(edges.src) | set(edges.dst)).issubset(set(nodes.gid)))
        for row in clusters.itertuples():
            members = nodes.loc[nodes.cluster_id == row.cluster_id]
            self.assertEqual(len(members), row.n_nodes)
            internal = edges.loc[edges.src.isin(members.gid) & edges.dst.isin(members.gid), "sum_kzt"].sum()
            self.assertEqual(internal, row.sum_kzt_internal)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        self.original_results = app.RESULTS_DIR
        self.original_mock = app.MOCK_DIR
        st.cache_data.clear()
        self.script = f"import app\nfrom pathlib import Path\napp.RESULTS_DIR = Path({str(self.output)!r})\napp.main()"

    def tearDown(self):
        st.cache_data.clear()
        app.RESULTS_DIR = self.original_results
        app.MOCK_DIR = self.original_mock
        self.temp.cleanup()

    def launch(self):
        result = AppTest.from_string(self.script, default_timeout=40).run()
        self.assertEqual([e.message for e in result.exception], [])
        return result

    def choose_source(self, at, source):
        next(r for r in at.radio if r.label == "Источник данных").set_value(source)
        at.run()
        self.assertFalse(at.exception)
        return at

    def write_results(self, nodes=None):
        mock_nodes, clusters, top, edges = app.demo_results()
        nodes = mock_nodes if nodes is None else nodes
        for filename, table in [("nodes_roles.csv", nodes), ("clusters.csv", clusters),
                                ("top_nodes.csv", top.iloc[::-1]), ("edges.csv", edges)]:
            table.to_csv(self.output / filename, index=False)
        return nodes, clusters, top, edges

    def test_top_navigation_and_back_return_to_dashboard_without_losing_gid(self):
        at = self.launch()
        initial = at.session_state.traceflow_active_gid
        at.selectbox(key="all_top_gid").set_value("902188")
        at.button(key="all_top_open").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state.traceflow_active_gid, "902188")
        self.assertEqual(at.session_state.traceflow_tab, "Dashboard")
        next(b for b in at.button if b.label == "← Предыдущий GID").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state.traceflow_active_gid, initial)
        self.assertEqual(at.session_state.traceflow_history, [])

    def test_direction_filter_and_graph_navigation(self):
        at = self.launch()
        direction = next(r for r in at.radio if r.label == "Направление связей")
        direction.set_value("Входящие").run()
        self.assertFalse(at.exception)
        peer = next(s for s in at.selectbox if s.label == "Связанный GID")
        self.assertEqual(set(peer.options), {"902188", "903470"})
        expected = peer.value
        next(b for b in at.button if b.label == "Продолжить по связи →").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state.traceflow_active_gid, expected)

    def test_empty_direction_preserves_other_observed_connections(self):
        at = self.launch()
        at.text_input[0].set_value("908001")
        next(b for b in at.button if b.label == "Открыть профиль").click().run()
        next(r for r in at.radio if r.label == "Направление связей").set_value("Входящие").run()
        self.assertFalse(at.exception)
        self.assertTrue(any("Всего у GID 908001: 1 связей" in i.value for i in at.info))
        next(r for r in at.radio if r.label == "Направление связей").set_value("Исходящие").run()
        self.assertTrue(any(s.label == "Связанный GID" for s in at.selectbox))

    def test_pipeline_rank_is_not_recomputed_from_priority_score(self):
        _, _, top, _ = self.write_results()
        # The team may use tie-breaking or analyst-reviewed ranking.
        top.loc[top.gid.eq("902188"), "rank"] = 40
        top.loc[top.gid.eq("901245"), "rank"] = 50
        top.to_csv(self.output / "top_nodes.csv", index=False)
        at = self.launch()
        self.assertEqual(at.session_state.traceflow_active_gid, "904311")
        tables = " ".join(m.value for m in at.markdown if "trace-table" in m.value)
        self.assertIn("#40", tables)
        self.assertIn("#50", tables)

    def test_broken_edges_show_unavailable_instead_of_isolated_node(self):
        self.write_results()
        (self.output / "edges.csv").write_text("wrong,columns\n1,2\n", encoding="utf-8")
        at = self.launch()
        self.assertTrue(any("src и dst" in w.value for w in at.warning))
        self.assertFalse(any("нет переводов этого клиента" in i.value for i in at.info))
        self.assertFalse(any(m.label == "Получено в графе" for m in at.metric))

    def test_default_demo_does_not_need_analytics_or_parquet(self):
        with patch("app.read_source", side_effect=AssertionError("Demo must not read real data")):
            at = self.launch()
        self.assertTrue(any("ДЕМО-РЕЖИМ" in m.value for m in at.markdown))
        self.assertEqual([tab.label for tab in at.tabs][:2], ["Dashboard", "Top Nodes"])
        self.assertEqual(at.session_state.traceflow_active_gid, "901245")
        card = next(m.value for m in at.markdown if 'class="subject-id"' in m.value)
        self.assertIn("Обоснование роли", card)
        self.assertIn("Причина приоритета", card)
        self.assertIn("5 930 000", card)

    def test_real_data_without_pipeline_is_explicitly_unscored(self):
        at = self.choose_source(self.launch(), "Исходные данные")
        self.assertTrue(any("Открыты исходные данные" in item.value for item in at.info))
        self.assertIn("Каталог клиентов", " ".join(m.value for m in at.markdown))

    def test_search_unknown_then_isolated_client(self):
        at = self.choose_source(self.launch(), "Исходные данные")
        at.text_input[0].set_value("does-not-exist")
        next(b for b in at.button if b.label == "Открыть профиль").click()
        at.run()
        self.assertTrue(any("Клиент не найден" in w.value for w in at.warning))
        nodes = pd.read_parquet(ROOT / "data (1)/data/nodes.parquet")
        edges = pd.read_parquet(ROOT / "data (1)/data/edges.parquet")
        orphan = next(iter(set(nodes.gid) - set(edges.src) - set(edges.dst)))
        at.text_input[0].set_value(str(orphan))
        next(b for b in at.button if b.label == "Открыть профиль").click()
        at.run()
        self.assertEqual(at.session_state.traceflow_active_gid, str(orphan))
        self.assertFalse(at.exception)
        self.assertTrue(any("нет переводов" in item.value for item in at.info))

    def test_search_real_seed_and_browser_tables_use_string_ids(self):
        at = self.choose_source(self.launch(), "Исходные данные")
        source = pd.read_parquet(ROOT / "data (1)/data/nodes.parquet")
        edges = pd.read_parquet(ROOT / "data (1)/data/edges.parquet")
        gid = str(source.loc[source.is_seed & source.gid.isin(edges.src), "gid"].iloc[0])
        at.text_input[0].set_value(gid)
        next(b for b in at.button if b.label == "Открыть профиль").click()
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state.traceflow_active_gid, gid)
        self.assertTrue(any("входящий поток неполон" in item.value for item in at.info))
        checked = set()
        for widget in at.dataframe:
            for column in ("gid", "GID", "Контрагент", "Отправитель", "Получатель"):
                if column in widget.value:
                    checked.add(column)
                    self.assertTrue(widget.value[column].map(lambda value: isinstance(value, str)).all())
        self.assertTrue({"gid", "Отправитель", "Получатель"}.issubset(checked))

    def test_depth_four_caveat_is_visible_for_real_boundary_node(self):
        at = self.choose_source(self.launch(), "Исходные данные")
        source = pd.read_parquet(ROOT / "data (1)/data/nodes.parquet")
        edges = pd.read_parquet(ROOT / "data (1)/data/edges.parquet")
        gid = str(source.loc[(source.depth == 4) & ~source.gid.isin(edges.src), "gid"].iloc[0])
        at.text_input[0].set_value(gid)
        next(b for b in at.button if b.label == "Открыть профиль").click()
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state.traceflow_active_gid, gid)
        self.assertTrue(any("Граница 4-го колена" in item.value for item in at.warning))

    def test_legacy_priority_scale_is_displayed_without_recalculation(self):
        nodes, _, top, _ = self.write_results()
        nodes.priority_score *= 100
        top.priority_score *= 100
        top["why"] = "Готовая причина аналитики; шкала /100."
        nodes.to_csv(self.output / "nodes_roles.csv", index=False)
        top.to_csv(self.output / "top_nodes.csv", index=False)
        at = self.launch()
        card = next(m.value for m in at.markdown if 'class="subject-id"' in m.value)
        self.assertIn("Приоритет / 100", card)
        self.assertIn("98.00", card)
        self.assertTrue(any("без пересчёта" in item.value for item in at.warning))

    def test_long_evidence_warns_without_blocking_search_or_truncating_text(self):
        nodes, _, _, _ = self.write_results()
        full_text = "Обоснование аналитики: " + "наблюдение " * 20
        nodes.loc[nodes.gid == "901245", "evidence"] = full_text
        nodes.to_csv(self.output / "nodes_roles.csv", index=False)
        at = self.launch()
        self.assertTrue(at.text_input)
        card = next(m.value for m in at.markdown if 'class="subject-id"' in m.value)
        self.assertIn(full_text, card)
        self.assertTrue(any("длиннее 200" in item.value for item in at.warning))

    def test_design_preview_is_separate_and_can_close(self):
        at = self.launch()
        self.assertTrue(any("ДЕМО-РЕЖИМ" in item.value for item in at.markdown))
        self.choose_source(at, "Исходные данные")
        self.assertTrue(any("Открыты исходные данные" in item.value for item in at.info))
        self.choose_source(at, "Демо: mock CSV")
        self.assertEqual(at.session_state.traceflow_active_gid, "901245")

    def test_pipeline_and_invalid_export(self):
        nodes, _, _, _ = self.write_results()
        at = self.launch()
        self.assertEqual(at.session_state.traceflow_active_gid, "901245")
        self.assertFalse(any("ДЕМО-РЕЖИМ" in m.value for m in at.markdown))
        nodes.loc[0, "role_score"] = 5
        nodes.to_csv(self.output / "nodes_roles.csv", index=False)
        st.cache_data.clear()
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("от 0 до 1" in m.value for m in at.markdown))

    def test_any_mock_gid_is_searchable_with_role_and_priority_reason(self):
        at = self.launch()
        nodes, _, top, _ = app.demo_results()
        for node in nodes.itertuples():
            with self.subTest(gid=node.gid):
                at.text_input[0].set_value(node.gid)
                next(b for b in at.button if b.label == "Открыть профиль").click()
                at.run()
                self.assertFalse(at.exception)
                self.assertEqual(at.session_state.traceflow_active_gid, node.gid)
                card = next(m.value for m in at.markdown if 'class="subject-id"' in m.value)
                self.assertIn(node.evidence, card)
                reason = top.loc[top.gid.eq(node.gid), "why"]
                self.assertIn(reason.iloc[0] if not reason.empty else node.priority_reason, card)

    def test_refresh_switches_mock_to_output_without_changing_exports(self):
        at = self.launch()
        nodes, _, top, _ = self.write_results()
        nodes.loc[nodes.gid == "901245", "evidence"] = "Уникальное объяснение роли от аналитики: 7 связей."
        nodes.to_csv(self.output / "nodes_roles.csv", index=False)
        top.loc[top.gid == "901245", "why"] = "Уникальное объяснение приоритета от аналитики: 2 ветки."
        top.to_csv(self.output / "top_nodes.csv", index=False)
        before = {p.name: p.read_bytes() for p in self.output.glob("*.csv")}
        next(b for b in at.button if b.label == "Обновить данные").click()
        at.run()
        self.assertFalse(at.exception)
        self.assertFalse(any("ДЕМО-РЕЖИМ" in m.value for m in at.markdown))
        card = next(m.value for m in at.markdown if 'class="subject-id"' in m.value)
        self.assertIn("Уникальное объяснение роли", card)
        self.assertIn("Уникальное объяснение приоритета", card)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.output.glob("*.csv")})

    def test_partial_or_empty_output_does_not_silently_use_mock(self):
        nodes, _, top, _ = app.demo_results()
        top.to_csv(self.output / "top_nodes.csv", index=False)
        at = self.launch()
        self.assertTrue(any("Не найден nodes_roles.csv" in m.value for m in at.markdown))
        nodes.iloc[:0].to_csv(self.output / "nodes_roles.csv", index=False)
        st.cache_data.clear()
        at.run()
        self.assertTrue(any("пока нет результатов" in m.value for m in at.markdown))
        self.assertFalse(any("ДЕМО-РЕЖИМ" in m.value for m in at.markdown))
        self.choose_source(at, "Демо: mock CSV")
        self.assertTrue(any("ДЕМО-РЕЖИМ" in m.value for m in at.markdown))

    def test_node_outside_top_without_priority_reason_is_honest(self):
        nodes, _, _, _ = self.write_results()
        nodes = nodes.drop(columns="priority_reason")
        nodes.to_csv(self.output / "nodes_roles.csv", index=False)
        at = self.launch()
        at.text_input[0].set_value("909999")
        next(b for b in at.button if b.label == "Открыть профиль").click()
        at.run()
        self.assertFalse(at.exception)
        card = next(m.value for m in at.markdown if 'class="subject-id"' in m.value)
        self.assertIn("Отдельное обоснование приоритета не передано", card)

    def test_long_gid_from_output_is_exactly_searchable(self):
        nodes, clusters, top, edges = app.demo_results()
        exact_gid = "9223372036854775807"
        for table, columns in [(nodes, ["gid"]), (top, ["gid"]), (edges, ["src", "dst"])]:
            for column in columns:
                table[column] = table[column].replace("901245", exact_gid)
        for name, table in [("nodes_roles.csv", nodes), ("clusters.csv", clusters), ("top_nodes.csv", top), ("edges.csv", edges)]:
            table.to_csv(self.output / name, index=False)
        at = self.launch()
        at.text_input[0].set_value(exact_gid)
        next(b for b in at.button if b.label == "Открыть профиль").click()
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state.traceflow_active_gid, exact_gid)

    def test_neighbor_navigation(self):
        at = self.launch()
        peer = next((s for s in at.selectbox if s.label == "Перейти к связанному клиенту"), None)
        if peer is None:
            self.fail("Default real client must expose its observed peers")
        expected = peer.value
        next(b for b in at.button if b.label == "Открыть контрагента").click()
        at.run()
        self.assertFalse(at.exception)
        self.assertEqual(at.session_state.traceflow_active_gid, expected)


if __name__ == "__main__":
    unittest.main()
