"""Regression checks for the judge's search and evidence workflow."""
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ui"))

import pandas as pd
import streamlit as st
from streamlit.testing.v1 import AppTest
import app
from components import normalise_id
from graph_view import _network_canvas


class ContractTests(unittest.TestCase):
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
        nodes.loc[0, "gid"] = 901245
        html = _network_canvas("<script>alert(1)</script>", edges.head(1), nodes, False)
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("marker-end", html)
        self.assertIn("zoom-reset", html)


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name)
        st.cache_data.clear()
        self.script = f"import app\nfrom pathlib import Path\napp.RESULTS_DIR = Path({str(self.output)!r})\napp.main()"

    def tearDown(self):
        st.cache_data.clear()
        self.temp.cleanup()

    def launch(self):
        result = AppTest.from_string(self.script, default_timeout=40).run()
        self.assertEqual([e.message for e in result.exception], [])
        return result

    def test_real_data_without_pipeline_is_explicitly_unscored(self):
        at = self.launch()
        self.assertTrue(any("Открыты исходные данные" in item.value for item in at.info))
        self.assertIn("Каталог клиентов", " ".join(m.value for m in at.markdown))

    def test_search_unknown_then_isolated_client(self):
        at = self.launch()
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

    def test_design_preview_is_separate_and_can_close(self):
        at = self.launch()
        next(b for b in at.button if b.label == "Открыть дизайн-кейс").click()
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("ДЕМО-РЕЖИМ" in item.value for item in at.markdown))
        next(b for b in at.button if b.label == "Закрыть дизайн-кейс").click()
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("Открыты исходные данные" in item.value for item in at.info))

    def test_pipeline_and_invalid_export(self):
        nodes, clusters, top, edges = app.demo_results()
        # Deliberately shuffled queue: the UI must restore score ordering.
        for filename, table in [("nodes_roles.csv", nodes), ("clusters.csv", clusters),
                                ("top_nodes.csv", top.iloc[::-1]), ("edges.csv", edges)]:
            table.to_csv(self.output / filename, index=False)
        at = self.launch()
        self.assertEqual(at.session_state.traceflow_active_gid, "901245")
        self.assertTrue(any("требуется не менее 20" in w.value for w in at.warning))
        nodes.loc[0, "role_score"] = 5
        nodes.to_csv(self.output / "nodes_roles.csv", index=False)
        st.cache_data.clear()
        at.run()
        self.assertFalse(at.exception)
        self.assertTrue(any("от 0 до 1" in m.value for m in at.markdown))

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
