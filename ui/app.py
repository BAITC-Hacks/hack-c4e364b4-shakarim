"""Streamlit interface for reviewing precomputed financial-network results.

Run from the repository root:
    streamlit run ui/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from components import inject_styles, normalise_id, render_brand, render_client_card, render_cluster, render_role_legend, render_top_nodes
from graph_view import load_edges, render_client_connections


ROOT = Path(__file__).resolve().parents[1]
# After the analytics branch is integrated, change only this line to ROOT / "output".
RESULTS_DIR = ROOT / "mock"


def read_result(filename: str) -> tuple[pd.DataFrame, str | None]:
    """Read a prepared result and turn filesystem/format failures into UI errors."""
    path = RESULTS_DIR / filename
    if not path.exists():
        return pd.DataFrame(), f"Файл не найден: {path}"
    try:
        return pd.read_csv(path, encoding="utf-8-sig"), None
    except Exception as error:
        return pd.DataFrame(), f"Не удалось прочитать {path.name}: {error}"


def find_selected_client(nodes: pd.DataFrame, requested_gid: str, fallback_gid: object) -> pd.Series:
    if nodes.empty or "gid" not in nodes.columns:
        return pd.Series(dtype=object)
    keys = nodes["gid"].map(normalise_id)
    if requested_gid.strip():
        matched = nodes.loc[keys == normalise_id(requested_gid)]
        if not matched.empty:
            return matched.iloc[0]
        st.warning(f"gid {requested_gid.strip()} не найден. Показан выбранный клиент из списка.")
    matched = nodes.loc[keys == normalise_id(fallback_gid)]
    return matched.iloc[0] if not matched.empty else pd.Series(dtype=object)


def main() -> None:
    st.set_page_config(page_title="TraceFlow | Аналитик", page_icon="◈", layout="wide")
    inject_styles()
    with st.sidebar:
        st.markdown('<div class="tf-sidebar-label">Workspace</div>', unsafe_allow_html=True)
        render_brand()
    st.markdown(
        '<div class="tf-header"><div><div class="tf-kicker">Financial network intelligence</div><h1>Investigation workspace</h1><div class="tf-subtitle">Разбор структуры переводов по готовым аналитическим результатам</div></div><div class="tf-status">● PIPELINE READY</div></div>',
        unsafe_allow_html=True,
    )

    nodes, nodes_error = read_result("nodes_roles.csv")
    clusters, clusters_error = read_result("clusters.csv")
    top_nodes, top_nodes_error = read_result("top_nodes.csv")
    if nodes.empty:
        st.error(nodes_error or f"Файл nodes_roles.csv пуст: {RESULTS_DIR}")
        st.stop()
    if clusters_error:
        st.warning(clusters_error)
    if top_nodes_error:
        st.warning(top_nodes_error)

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

    st.sidebar.markdown('<div class="tf-sidebar-label">Find a client</div>', unsafe_allow_html=True)
    requested_gid = st.sidebar.text_input("Введите gid", placeholder="Например, 12345")
    options = nodes["gid"].tolist()
    default_index = 0
    if not top_nodes.empty and "gid" in top_nodes.columns:
        top_gid = normalise_id(top_nodes.iloc[0]["gid"])
        default_index = next((index for index, gid in enumerate(options) if normalise_id(gid) == top_gid), 0)
    fallback_gid = st.sidebar.selectbox("Или выберите из списка", options, index=default_index, format_func=str)
    with st.sidebar:
        st.divider()
        st.markdown('<div class="tf-sidebar-label">Role legend</div>', unsafe_allow_html=True)
        render_role_legend()
        if edge_path:
            st.caption(f"Связи: {edge_path.relative_to(ROOT)}")

    selected = find_selected_client(nodes, requested_gid, fallback_gid)
    if selected.empty:
        st.error("Не удалось выбрать клиента из nodes_roles.csv.")
        st.stop()
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
