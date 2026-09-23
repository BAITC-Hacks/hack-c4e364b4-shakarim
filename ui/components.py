"""Reusable, presentation-focused Streamlit components for the analyst UI."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st


ROLE_COLORS = {
    "consolidator": "#7C3AED",
    "transit": "#2563EB",
    "distributor": "#EA580C",
    "terminal": "#16A34A",
    "coordinator": "#DC2626",
    "peripheral": "#64748B",
}
DEFAULT_ROLE_COLOR = "#94A3B8"


def role_color(role: Any) -> str:
    """Return a stable colour for a pipeline-provided role."""
    return ROLE_COLORS.get(str(role).strip().lower(), DEFAULT_ROLE_COLOR)


def format_score(value: Any) -> str:
    """Format a scalar without turning missing pipeline values into a score."""
    if pd.isna(value):
        return "—"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def role_badge(role: Any) -> str:
    label = str(role).strip() if pd.notna(role) and str(role).strip() else "not assigned"
    return (
        f'<span style="background:{role_color(role)};color:white;border-radius:999px;'
        f'padding:0.28rem 0.65rem;font-weight:650;font-size:0.9rem">{label}</span>'
    )


def render_role_legend() -> None:
    badges = " ".join(role_badge(role) for role in ROLE_COLORS)
    st.markdown(f"<div style='line-height:2.4'>{badges}</div>", unsafe_allow_html=True)


def render_client_card(client: pd.Series) -> None:
    """Show only fields prepared by the analytics pipeline."""
    st.subheader(f"Клиент {client['gid']}")
    st.markdown(role_badge(client.get("role")), unsafe_allow_html=True)

    left, middle, right = st.columns(3)
    left.metric("Role score", format_score(client.get("role_score")))
    middle.metric("Priority score", format_score(client.get("priority_score")))
    right.metric("Cluster", str(client.get("cluster_id", "—")))

    evidence = client.get("evidence")
    st.caption("Объяснение от аналитического пайплайна")
    st.info(str(evidence) if pd.notna(evidence) and str(evidence).strip() else "Нет объяснения в выгрузке.")


def render_top_nodes(top_nodes: pd.DataFrame) -> None:
    st.subheader("TOP-20 узлов")
    if top_nodes.empty:
        st.info("Файл top_nodes.csv пока не содержит строк.")
        return

    visible = top_nodes.head(20).copy()
    wanted = [column for column in ["rank", "gid", "role", "priority_score", "why"] if column in visible]
    st.dataframe(visible[wanted], hide_index=True, use_container_width=True)


def render_cluster(client: pd.Series, nodes: pd.DataFrame, clusters: pd.DataFrame) -> None:
    cluster_id = client.get("cluster_id")
    st.subheader("Кластер клиента")
    if pd.isna(cluster_id):
        st.info("Для клиента не указан кластер.")
        return

    cluster_row = clusters.loc[clusters["cluster_id"].astype(str) == str(cluster_id)] if "cluster_id" in clusters else pd.DataFrame()
    members = nodes.loc[nodes["cluster_id"].astype(str) == str(cluster_id)] if "cluster_id" in nodes else pd.DataFrame()

    if not cluster_row.empty:
        details = cluster_row.iloc[0]
        first, second, third = st.columns(3)
        first.metric("Узлов", str(details.get("n_nodes", len(members))))
        second.metric("Seed-клиентов", str(details.get("n_seed", "—")))
        internal_turnover = details.get("sum_kzt_internal", "—")
        third.metric("Внутренний оборот", str(internal_turnover))
        hypothesis = details.get("hypothesis")
        if pd.notna(hypothesis) and str(hypothesis).strip():
            st.caption("Гипотеза назначения")
            st.write(str(hypothesis))

    if members.empty:
        st.info("Состав кластера отсутствует в nodes_roles.csv.")
        return

    visible = members[[column for column in ["gid", "role", "role_score", "priority_score"] if column in members]].copy()
    st.caption(f"Показано участников: {len(visible)}")
    st.dataframe(visible, hide_index=True, use_container_width=True)
