"""Reusable, presentation-focused Streamlit components for the analyst UI."""

from __future__ import annotations

from html import escape
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


def normalise_id(value: Any) -> str:
    """Make CSV ids comparable when one file parsed them as int and another as float."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            pass
    return text


def inject_styles() -> None:
    """Apply the product visual system without adding a frontend dependency."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --ink:#172033; --muted:#68758a; --line:#e6eaf1; --surface:#ffffff; --accent:#315efb; }
        html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
        [data-testid="stAppViewContainer"] { background:#f5f7fb; }
        [data-testid="stHeader"] { background:rgba(245,247,251,.85); }
        .block-container { max-width:1440px; padding:2.25rem 3.25rem 4rem; }
        h1, h2, h3 { font-family:'Space Grotesk', sans-serif; color:var(--ink); letter-spacing:-.025em; }
        h1 { font-size:2.15rem !important; margin-bottom:.15rem !important; }
        h2 { font-size:1.25rem !important; }
        h3 { font-size:1.05rem !important; }
        [data-testid="stSidebar"] { background:#111827; border-right:0; }
        [data-testid="stSidebar"] * { color:#e5e7eb; }
        [data-testid="stSidebar"] input { background:#1f2937; border:1px solid #374151; color:#fff; }
        [data-testid="stSidebar"] [data-baseweb="select"] > div { background:#1f2937; border-color:#374151; }
        .tf-brand { display:flex; align-items:center; gap:.75rem; margin:.25rem 0 2.2rem; }
        .tf-mark { width:34px; height:34px; border-radius:10px; display:grid; place-items:center; background:#315efb; color:#fff; font-weight:700; box-shadow:0 7px 16px #315efb40; }
        .tf-brand-name { color:#fff; font:700 1.05rem 'Space Grotesk', sans-serif; }
        .tf-brand-sub { color:#94a3b8; font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; }
        .tf-header { display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:1.7rem; }
        .tf-kicker { color:#315efb; font-size:.72rem; font-weight:700; letter-spacing:.13em; text-transform:uppercase; margin-bottom:.4rem; }
        .tf-subtitle { color:var(--muted); font-size:.95rem; }
        .tf-status { color:#1b7b4b; background:#e6f7ee; border:1px solid #c6ecd8; border-radius:999px; padding:.42rem .7rem; font-size:.75rem; font-weight:700; letter-spacing:.04em; }
        .tf-panel { background:var(--surface); border:1px solid var(--line); border-radius:18px; padding:1.25rem 1.35rem; box-shadow:0 10px 30px #24324a08; }
        .tf-panel-title { color:var(--ink); font:600 1rem 'Space Grotesk', sans-serif; margin-bottom:.9rem; }
        .tf-eyebrow { color:#8190a6; font-size:.69rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; margin:1.25rem 0 .55rem; }
        .tf-metric { background:#f8fafc; border:1px solid #edf0f5; border-radius:13px; padding:.75rem .85rem; min-height:74px; }
        .tf-metric-label { color:#7a8799; font-size:.72rem; margin-bottom:.25rem; }
        .tf-metric-value { color:var(--ink); font:700 1.22rem 'Space Grotesk', sans-serif; }
        .tf-evidence { border-left:3px solid #315efb; background:#f4f7ff; border-radius:0 12px 12px 0; color:#35445a; padding:.8rem .95rem; font-size:.89rem; line-height:1.5; }
        .tf-empty { border:1px dashed #cbd5e1; border-radius:14px; padding:1.1rem; color:#758399; background:#fafbfc; }
        .tf-sidebar-label { color:#94a3b8; font-size:.69rem; font-weight:700; letter-spacing:.11em; text-transform:uppercase; margin:1rem 0 .55rem; }
        div[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand() -> None:
    st.markdown(
        '<div class="tf-brand"><div class="tf-mark">◈</div><div><div class="tf-brand-name">TraceFlow</div><div class="tf-brand-sub">Network intelligence</div></div></div>',
        unsafe_allow_html=True,
    )


def render_pipeline_waiting_state(message: str) -> None:
    """Keep the product shell useful while analytical exports are being generated."""
    st.markdown(
        f'<div class="tf-panel" style="margin-top:2rem;max-width:760px">'
        '<div class="tf-eyebrow">Pipeline status</div>'
        '<div class="tf-panel-title" style="font-size:1.45rem">Waiting for analytical results</div>'
        f'<div class="tf-evidence">{escape(message)}</div>'
        '<div style="color:#68758a;font-size:.9rem;margin-top:1rem;line-height:1.55">'
        'Интерфейс готов. После появления nodes_roles.csv, clusters.csv и top_nodes.csv '
        'данные подгрузятся при следующем обновлении страницы.</div></div>',
        unsafe_allow_html=True,
    )


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
    label = escape(label)
    return (
        f'<span style="background:{role_color(role)};color:white;border-radius:999px;'
        f'padding:0.28rem 0.65rem;font-weight:650;font-size:0.9rem">{label}</span>'
    )


def render_role_legend() -> None:
    badges = " ".join(role_badge(role) for role in ROLE_COLORS)
    st.markdown(f"<div style='line-height:2.4'>{badges}</div>", unsafe_allow_html=True)


def render_client_card(client: pd.Series) -> None:
    """Show only fields prepared by the analytics pipeline."""
    st.markdown('<div class="tf-eyebrow">Selected subject</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="tf-panel-title" style="font-size:1.35rem">Client {escape(str(client["gid"]))}</div>', unsafe_allow_html=True)
    st.markdown(role_badge(client.get("role")), unsafe_allow_html=True)

    left, middle, right = st.columns(3)
    left.markdown(f'<div class="tf-metric"><div class="tf-metric-label">Role score</div><div class="tf-metric-value">{format_score(client.get("role_score"))}</div></div>', unsafe_allow_html=True)
    middle.markdown(f'<div class="tf-metric"><div class="tf-metric-label">Priority score</div><div class="tf-metric-value">{format_score(client.get("priority_score"))}</div></div>', unsafe_allow_html=True)
    right.markdown(f'<div class="tf-metric"><div class="tf-metric-label">Cluster</div><div class="tf-metric-value">{client.get("cluster_id", "—")}</div></div>', unsafe_allow_html=True)

    evidence = client.get("evidence")
    st.markdown('<div class="tf-eyebrow">Evidence from pipeline</div>', unsafe_allow_html=True)
    text = str(evidence) if pd.notna(evidence) and str(evidence).strip() else "Нет объяснения в выгрузке."
    st.markdown(f'<div class="tf-evidence">{escape(text)}</div>', unsafe_allow_html=True)


def render_top_nodes(top_nodes: pd.DataFrame) -> None:
    st.markdown('<div class="tf-eyebrow">Prioritised review queue</div>', unsafe_allow_html=True)
    st.markdown('<div class="tf-panel-title" style="font-size:1.25rem">TOP-20 nodes</div>', unsafe_allow_html=True)
    if top_nodes.empty:
        st.info("Файл top_nodes.csv пока не содержит строк.")
        return

    visible = top_nodes.head(20).copy()
    wanted = [column for column in ["rank", "gid", "role", "priority_score", "why"] if column in visible]
    if not wanted:
        st.markdown('<div class="tf-empty">В top_nodes.csv нет отображаемых колонок.</div>', unsafe_allow_html=True)
        return
    st.dataframe(visible[wanted], hide_index=True, use_container_width=True)


def render_cluster(client: pd.Series, nodes: pd.DataFrame, clusters: pd.DataFrame) -> None:
    cluster_id = client.get("cluster_id")
    st.markdown('<div class="tf-eyebrow">Network context</div>', unsafe_allow_html=True)
    st.markdown('<div class="tf-panel-title">Client cluster</div>', unsafe_allow_html=True)
    if pd.isna(cluster_id):
        st.info("Для клиента не указан кластер.")
        return

    cluster_key = normalise_id(cluster_id)
    cluster_row = clusters.loc[clusters["cluster_id"].map(normalise_id) == cluster_key] if "cluster_id" in clusters else pd.DataFrame()
    members = nodes.loc[nodes["cluster_id"].map(normalise_id) == cluster_key] if "cluster_id" in nodes else pd.DataFrame()

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
