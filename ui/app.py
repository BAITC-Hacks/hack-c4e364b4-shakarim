"""Streamlit interface for reviewing precomputed financial-network results.

Run from the repository root:
    streamlit run ui/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from components import render_client_card, render_cluster, render_role_legend, render_top_nodes
from graph_view import load_edges, render_client_connections


ROOT = Path(__file__).resolve().parents[1]
# After the analytics branch is integrated, change only this line to ROOT / "output".
RESULTS_DIR = ROOT / "mock"


@st.cache_data(show_spinner=False)
def read_result(filename: str) -> pd.DataFrame:
    path = RESULTS_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def find_selected_client(nodes: pd.DataFrame, requested_gid: str, fallback_gid: object) -> pd.Series:
    if requested_gid.strip():
        matched = nodes.loc[nodes["gid"].astype(str) == requested_gid.strip()]
        if not matched.empty:
            return matched.iloc[0]
        st.warning(f"gid {requested_gid.strip()} не найден. Показан выбранный клиент из списка.")
    return nodes.loc[nodes["gid"].astype(str) == str(fallback_gid)].iloc[0]


def main() -> None:
    st.set_page_config(page_title="TraceFlow | Аналитик", page_icon="◈", layout="wide")
    st.title("TraceFlow")
    st.caption("Экран аналитика: готовые роли, приоритеты и доказательства из пайплайна.")

    nodes = read_result("nodes_roles.csv")
    clusters = read_result("clusters.csv")
    top_nodes = read_result("top_nodes.csv")
    if nodes.empty:
        st.error(f"Не найден или пуст {RESULTS_DIR / 'nodes_roles.csv'}. Добавьте результаты аналитического пайплайна.")
        st.stop()

    required = {"gid", "role", "role_score", "priority_score", "cluster_id", "evidence"}
    missing = sorted(required - set(nodes.columns))
    if missing:
        st.error(f"nodes_roles.csv не соответствует контракту: отсутствуют {', '.join(missing)}.")
        st.stop()

    edge_candidates = [
        RESULTS_DIR / "edges.parquet",
        RESULTS_DIR / "edges.csv",
        ROOT / "mock" / "edges.parquet",
        ROOT / "mock" / "edges.csv",
        ROOT / "data" / "edges.parquet",
        ROOT / "data (1)" / "data" / "edges.parquet",
    ]
    edges, edge_path = load_edges(edge_candidates)

    st.sidebar.header("Поиск клиента")
    requested_gid = st.sidebar.text_input("Введите gid", placeholder="Например, 12345")
    options = nodes["gid"].tolist()
    default_index = 0
    if not top_nodes.empty and "gid" in top_nodes.columns:
        top_gid = str(top_nodes.iloc[0]["gid"])
        default_index = next((index for index, gid in enumerate(options) if str(gid) == top_gid), 0)
    fallback_gid = st.sidebar.selectbox("Или выберите из списка", options, index=default_index, format_func=str)
    st.sidebar.divider()
    st.sidebar.caption("Цвета ролей")
    render_role_legend()
    if edge_path:
        st.sidebar.caption(f"Связи: {edge_path.relative_to(ROOT)}")

    selected = find_selected_client(nodes, requested_gid, fallback_gid)
    left, right = st.columns([1.05, 1.55], gap="large")
    with left:
        render_client_card(selected)
        render_cluster(selected, nodes, clusters)
    with right:
        render_client_connections(selected["gid"], edges, nodes)

    st.divider()
    render_top_nodes(top_nodes)


if __name__ == "__main__":
    main()
