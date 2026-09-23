"""Visual system and read-only UI components for the TraceFlow analyst product."""

from __future__ import annotations

from html import escape
from decimal import Decimal, InvalidOperation
from typing import Any

import pandas as pd
import streamlit as st


ROLE_COLORS = {
    "consolidator": "#B58CFF",
    "transit": "#47B4FF",
    "distributor": "#FF9F5A",
    "terminal": "#4DE2A8",
    "coordinator": "#FF6B9A",
    "peripheral": "#8090AB",
}
ROLE_LABELS = {
    "unassigned": "Роль не рассчитана",
    "consolidator": "Консолидатор",
    "transit": "Транзит",
    "distributor": "Распределитель",
    "terminal": "Получатель",
    "coordinator": "Координатор",
    "peripheral": "Периферия",
}
DEFAULT_ROLE_COLOR = "#8090AB"


def normalise_id(value: Any) -> str:
    """Make CSV identifiers stable across integer, float and string parsing."""
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(Decimal(text)))
        except (InvalidOperation, ValueError, OverflowError):
            pass
    return text


def role_color(role: Any) -> str:
    return ROLE_COLORS.get(str(role).strip().lower(), DEFAULT_ROLE_COLOR)


def role_label(role: Any) -> str:
    key = str(role).strip().lower()
    return ROLE_LABELS.get(key, str(role).strip() or "Не определена")


def format_score(value: Any, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return escape(str(value))


def format_money(value: Any) -> str:
    if value is None or pd.isna(value):
        return "—"
    try:
        return f"{float(value):,.0f}".replace(",", " ") + " ₸"
    except (TypeError, ValueError):
        return escape(str(value))


def role_badge(role: Any) -> str:
    color = role_color(role)
    label = escape(role_label(role))
    return f'<span class="role-badge" style="--role:{color}"><i></i>{label}</span>'


def open_client(gid: object) -> None:
    """Widget callback: all entry points open the same string-ID profile."""
    st.session_state.traceflow_active_gid = normalise_id(gid)
    st.session_state.traceflow_tab = "Dashboard"


def open_selected_client(widget_key: str) -> None:
    open_client(st.session_state[widget_key])


def return_to_previous_client() -> None:
    history = st.session_state.get("traceflow_history", [])
    if history:
        gid = history.pop()
        st.session_state.traceflow_last_gid = gid
        open_client(gid)


def investigation_brief(client: pd.Series, priority_reason: object, edges: pd.DataFrame,
                        source: str, priority_maximum: int = 1) -> str:
    """Portable evidence note; export supplied conclusions without re-scoring."""
    gid = normalise_id(client.gid)
    lines = ["# TraceFlow · сводка проверки", "", f"GID: {gid}", f"Источник: {source}",
             f"Роль (гипотеза): {role_label(client.get('role'))}",
             f"Оценка роли: {format_score(client.get('role_score'))}",
             f"Кластер: {normalise_id(client.get('cluster_id'))}",
             f"Приоритет: {format_score(client.get('priority_score'), 3 if priority_maximum == 1 else 2)} / {priority_maximum}",
             "", "## Обоснование роли из CSV", str(client.get("evidence", "Не передано")),
             "", "## Причина приоритета из CSV",
             str(priority_reason) if pd.notna(priority_reason) and str(priority_reason).strip() else "Не передана.",
             "", "## Наблюдаемые связи"]
    if edges.attrs.get("load_error"):
        lines.append("Связи недоступны: " + edges.attrs["load_error"])
    else:
        incident = edges.loc[edges.src.map(normalise_id).eq(gid) | edges.dst.map(normalise_id).eq(gid)]
        lines.append(f"Всего направленных связей: {len(incident)}. Сводка включает все страницы графа.")
        for row in incident.itertuples():
            lines.append(f"- {normalise_id(row.src)} → {normalise_id(row.dst)}; {format_money(getattr(row, 'sum_kzt', None))}")
    lines.extend(["", "## Ограничения", "Роль и кластер — гипотезы для проверки. Оценка роли не является вероятностью нарушения.",
                  "Июль 2026; внутрибанковские переводы от 5 000 ₸; обход по исходящим до 4 колен.",
                  "Отсутствие наблюдаемых связей не доказывает отсутствие активности вне выборки."])
    if str(client.get("is_seed", "")).lower() in {"true", "1", "1.0"}:
        lines.append("SEED: входящий поток неполон; сопоставление входа и выхода не отражает полный баланс.")
    if str(client.get("depth", "")) in {"4", "4.0"}:
        lines.append("DEPTH=4: граница выгрузки; дальнейшие переводы могут быть неизвестны.")
    return "\n".join(lines) + "\n"


def inject_styles() -> None:
    """Apply a self-contained premium dark product system without external assets."""
    st.markdown(
        """
        <style>
        :root {
          --bg:#080B13; --surface:#0F1523; --surface-2:#141C2D; --line:#253047;
          --text:#F3F6FF; --muted:#8F9BB3; --blue:#6B8CFF; --mint:#4DE2A8;
        }
        html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
        [data-testid="stAppViewContainer"] { background:radial-gradient(circle at 76% -12%, #1d2c673b 0, transparent 31%), var(--bg); color:var(--text); }
        [data-testid="stHeader"] { background:transparent; height:0; }
        [data-testid="stStatusWidget"], [data-testid="stToolbarActions"], [data-testid="stDeployButton"], #MainMenu, footer { display:none !important; }
        [data-testid="stHeader"]:has([data-testid="stExpandSidebarButton"]) { height:2.8rem; background:#0B101C; }
        [data-testid="stExpandSidebarButton"]::after { content:"Поиск GID"; font-size:.8rem; color:#EAF0FF; padding:0 .45rem; }
        [data-testid="stDecoration"] { display:none; }
        .block-container { max-width:1580px; padding:1.4rem 2.3rem 3.5rem; }
        [data-testid="stSidebar"] { background:#0B101C; border-right:1px solid #202a3d; }
        [data-testid="stSidebar"] > div:first-child { padding:1.3rem 1.15rem; }
        [data-testid="stSidebar"] * { color:var(--text); }
        [data-testid="stSidebar"] input, [data-testid="stSidebar"] [data-baseweb="select"] > div {
            background:#121A29 !important; border-color:#293652 !important; color:#EEF2FF !important;
        }
        [data-testid="stSidebar"] input:focus, [data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within { border-color:#728FFF !important; box-shadow:0 0 0 3px #6b8cff24 !important; }
        [data-testid="stSidebar"] .stButton > button, [data-testid="stSidebar"] .stFormSubmitButton > button { width:100%; min-height:2.55rem; border:1px solid #718BFF; border-radius:11px; background:linear-gradient(135deg,#6484FF,#5069E8); color:#F8FAFF; font-size:.79rem; font-weight:760; box-shadow:0 10px 24px #405ef744; transition:transform .2s ease,box-shadow .2s ease,filter .2s ease; }
        [data-testid="stSidebar"] .stButton > button:hover, [data-testid="stSidebar"] .stFormSubmitButton > button:hover { border-color:#A9B9FF; color:#FFF; filter:brightness(1.08); box-shadow:0 14px 30px #405ef766; transform:translateY(-1px); }
        [data-testid="stSidebar"] .stButton > button:active, [data-testid="stSidebar"] .stFormSubmitButton > button:active { transform:translateY(0); }
        .trace-brand { display:flex; align-items:center; gap:.78rem; padding:.2rem 0 1.65rem; border-bottom:1px solid #202a3d; margin-bottom:1.35rem; }
        .trace-brand svg { width:39px; height:39px; flex:0 0 auto; filter:drop-shadow(0 8px 16px #476dff55); }
        .trace-brand-name { color:#F7F9FF; font-size:1.08rem; font-weight:760; letter-spacing:-.04em; }
        .trace-brand-sub { color:#7F8DA9; font-size:.62rem; font-weight:700; letter-spacing:.17em; text-transform:uppercase; margin-top:.14rem; }
        .side-label { color:#7C8AA6; font-size:.65rem; font-weight:800; letter-spacing:.15em; text-transform:uppercase; margin:1.2rem 0 .48rem; }
        .side-caption { color:#687793; font-size:.75rem; line-height:1.48; margin-top:1.5rem; }
        .hero { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; padding:1.25rem 0 1.35rem; }
        .hero { flex-wrap:wrap; }
        [data-testid="stMetricValue"] { font-size:1.22rem; }
        [data-testid="stMetricValue"] > div { white-space:normal; overflow-wrap:anywhere; }
        .hero-kicker { color:#8299FF; font-size:.67rem; font-weight:800; letter-spacing:.18em; text-transform:uppercase; margin-bottom:.48rem; }
        .hero-title { color:#F5F7FF; font-size:2.15rem; font-weight:750; line-height:1.06; letter-spacing:-.06em; }
        .hero-copy { color:#94A1B9; font-size:.9rem; line-height:1.5; margin-top:.55rem; }
        .live-pill { display:inline-flex; align-items:center; gap:.42rem; border:1px solid #2D5F55; background:#12312988; color:#72E5B7; padding:.47rem .72rem; border-radius:999px; white-space:nowrap; font-size:.69rem; font-weight:800; letter-spacing:.09em; }
        .live-pill i { width:6px; height:6px; border-radius:50%; background:#4DE2A8; box-shadow:0 0 0 4px #4de2a822; animation:pulse 1.8s infinite; }
        .demo-pill { border-color:#6E5426; background:#342916aa; color:#FFD28B; }
        .demo-pill i { background:#FFB54D; box-shadow:0 0 0 4px #ffb54d22; }
        .stat-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.7rem; margin-bottom:1.25rem; }
        .stat-tile { background:linear-gradient(145deg,#121A2A,#0E1421); border:1px solid #222E45; border-radius:14px; padding:.85rem .95rem; }
        .stat-name { color:#8592AC; font-size:.67rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; }
        .stat-value { color:#F3F6FF; font-size:1.35rem; font-weight:720; letter-spacing:-.05em; margin-top:.28rem; }
        .stat-note { color:#6F7C95; font-size:.7rem; margin-top:.18rem; }
        .panel { background:linear-gradient(145deg,#111827eF,#0D1320f4); border:1px solid #253149; border-radius:18px; box-shadow:0 24px 50px #0000001f; animation:panel-in .38s cubic-bezier(.2,.8,.2,1) both; }
        .panel-head { display:flex; justify-content:space-between; align-items:center; padding:1rem 1.12rem .8rem; border-bottom:1px solid #222D42; }
        .panel-kicker { color:#7E8AA3; font-size:.63rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
        .panel-title { color:#F2F5FF; font-size:1.03rem; font-weight:720; letter-spacing:-.025em; margin-top:.22rem; }
        .panel-body { padding:1rem 1.12rem 1.12rem; }
        .subject-id { color:#F5F7FF; font-size:1.52rem; font-weight:760; letter-spacing:-.055em; margin-top:.35rem; }
        .subject-id { font-size:1.25rem; overflow-wrap:anywhere; }
        .panel-head { flex-wrap:wrap; gap:.55rem; }
        .role-badge { display:inline-flex; align-items:center; gap:.42rem; color:#EAF0FF; border:1px solid color-mix(in srgb, var(--role) 48%, transparent); background:color-mix(in srgb, var(--role) 12%, transparent); border-radius:999px; padding:.35rem .58rem; font-size:.73rem; font-weight:720; }
        .role-badge i { width:7px; height:7px; background:var(--role); border-radius:50%; box-shadow:0 0 12px var(--role); }
        .metric-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.55rem; margin-top:1rem; }
        .metric { background:#0B111D; border:1px solid #202C42; border-radius:12px; padding:.7rem; }
        .metric-label { color:#78869F; font-size:.62rem; font-weight:750; letter-spacing:.08em; text-transform:uppercase; }
        .metric-value { color:#F2F5FF; font-size:1.08rem; font-weight:720; letter-spacing:-.04em; margin-top:.3rem; }
        .evidence { margin-top:1rem; border-left:2px solid #6B8CFF; background:#101A2C; border-radius:0 12px 12px 0; padding:.85rem .9rem; color:#B8C4DA; font-size:.83rem; line-height:1.55; }
        .evidence-label { color:#7786A4; font-size:.62rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase; margin-bottom:.35rem; }
        .cluster-summary { display:grid; grid-template-columns:repeat(3,1fr); gap:.45rem; margin:.85rem 0; }
        .cluster-chip { background:#0C1320; border:1px solid #202C42; border-radius:10px; padding:.62rem; }
        .cluster-chip span { display:block; color:#74829B; font-size:.59rem; font-weight:800; letter-spacing:.07em; text-transform:uppercase; }
        .cluster-chip b { display:block; color:#EAF0FF; margin-top:.26rem; font-size:.83rem; }
        .cluster-hypothesis { color:#AAB6CD; font-size:.8rem; line-height:1.48; }
        .table-wrap { overflow:auto; border:1px solid #253149; border-radius:14px; background:#0D1320; }
        .trace-table { width:100%; border-collapse:collapse; min-width:680px; }
        .trace-table th { color:#71809A; font-size:.61rem; letter-spacing:.1em; text-transform:uppercase; text-align:left; padding:.75rem .8rem; background:#111A2A; border-bottom:1px solid #263149; }
        .trace-table td { color:#B8C5DA; padding:.78rem .8rem; border-bottom:1px solid #1D273A; font-size:.79rem; vertical-align:middle; }
        .trace-table tr:last-child td { border-bottom:none; }
        .trace-table tr:hover td { background:#151F31; }
        .rank { color:#7588FF; font-weight:800; }
        .quiet { color:#71809A; }
        .demo-banner { display:flex; align-items:center; gap:.7rem; border:1px solid #594827; background:#2C2518; border-radius:12px; padding:.72rem .85rem; color:#E9CC99; font-size:.8rem; margin-bottom:1rem; }
        .demo-banner b { color:#FFCC73; }
        .empty-state { padding:1.25rem; border:1px dashed #31415E; border-radius:14px; background:#0E1522; color:#92A0B9; font-size:.84rem; }
        .graph-intro { display:flex; justify-content:space-between; align-items:center; gap:1rem; margin-bottom:.62rem; padding:1rem 1.12rem .8rem; border:1px solid #253149; border-radius:18px; background:linear-gradient(145deg,#111827eF,#0D1320f4); }
        .graph-stage { border-radius:18px; overflow:hidden; box-shadow:0 24px 50px #0000001f; }
        [data-testid="stCustomComponentV1"], [data-testid="stIframe"] { margin-top:0 !important; }
        [data-testid="stCustomComponentV1"] iframe, [data-testid="stIframe"] iframe { border:0 !important; border-radius:18px; overflow:hidden; }
        [data-testid="stDataFrame"] { border:1px solid #253149; border-radius:13px; overflow:hidden; }
        [data-testid="stAlert"] { border-radius:12px; }
        @keyframes pulse { 50% { opacity:.45; transform:scale(.72); } }
        @keyframes panel-in { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
        @media(prefers-reduced-motion:reduce) { *, *::before, *::after { animation-duration:.01ms !important; animation-iteration-count:1 !important; scroll-behavior:auto !important; transition-duration:.01ms !important; } }
        @media(max-width:900px) { .block-container { padding:3rem 1rem 2.5rem; } .stat-grid { grid-template-columns:repeat(2,1fr); } .hero-title { font-size:1.7rem; } .metric-grid { grid-template-columns:1fr; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand() -> None:
    st.markdown(
        """
        <div class="trace-brand">
          <svg viewBox="0 0 48 48" fill="none" aria-label="TraceFlow logo">
            <defs><linearGradient id="traceflow-gradient" x1="5" y1="4" x2="43" y2="44" gradientUnits="userSpaceOnUse"><stop stop-color="#88A2FF"/><stop offset="1" stop-color="#4367FF"/></linearGradient></defs>
            <rect x="3" y="3" width="42" height="42" rx="13" fill="url(#traceflow-gradient)"/>
            <path d="M14 17.5 24 11l10 6.5v13L24 37l-10-6.5v-13Z" stroke="white" stroke-opacity=".92" stroke-width="1.6"/>
            <path d="M14 17.5 24 24l10-6.5M24 24v13" stroke="white" stroke-opacity=".72" stroke-width="1.4"/>
            <circle cx="24" cy="24" r="3.2" fill="white"/>
          </svg>
          <div><div class="trace-brand-name">TraceFlow</div><div class="trace-brand-sub">Финансовая разведка</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(demo_mode: bool, n_nodes: int, n_edges: int, n_top: int, raw_mode: bool = False) -> None:
    mode_class = "demo-pill" if demo_mode else ""
    mode_text = "ДЕМО-КЕЙС" if demo_mode else "ДАННЫЕ ЗАГРУЖЕНЫ"
    if raw_mode:
        mode_text = "ОБЗОР ИСХОДНЫХ ДАННЫХ"
    result_label = "Ожидает расчёта" if raw_mode else "Проверяемый"
    result_note = "роли ещё не переданы" if raw_mode else "выводы с evidence"
    nodes_label = f"{n_nodes:,}".replace(",", " ")
    edges_label = f"{n_edges:,}".replace(",", " ")
    top_label = f"{n_top:,}".replace(",", " ")
    st.markdown(
        f"""
        <div class="hero">
          <div><div class="hero-kicker">TraceFlow · финансовая разведка</div><div class="hero-title">Центр финансового расследования</div><div class="hero-copy">Единое рабочее пространство для проверки ролей, потоков и связей финансовой сети.</div></div>
          <div class="live-pill {mode_class}"><i></i>{mode_text}</div>
        </div>
        <div class="stat-grid">
          <div class="stat-tile"><div class="stat-name">Клиенты в контуре</div><div class="stat-value">{nodes_label}</div><div class="stat-note">в текущем кейсе</div></div>
          <div class="stat-tile"><div class="stat-name">Наблюдаемые связи</div><div class="stat-value">{edges_label}</div><div class="stat-note">направленных переводов</div></div>
          <div class="stat-tile"><div class="stat-name">Очередь проверки</div><div class="stat-value">{top_label}</div><div class="stat-note">приоритетных узлов</div></div>
          <div class="stat-tile"><div class="stat-name">Режим решения</div><div class="stat-value">{result_label}</div><div class="stat-note">{result_note}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_demo_banner(message: str) -> None:
    st.markdown(
        f'<div class="demo-banner"><b>ДЕМО-РЕЖИМ</b><span>{escape(message)}</span></div>',
        unsafe_allow_html=True,
    )


def render_role_legend() -> None:
    st.markdown('<div class="side-label">Легенда ролей</div>', unsafe_allow_html=True)
    badges = "<br>".join(role_badge(role) for role in ROLE_COLORS)
    st.markdown(f'<div style="display:grid;gap:.38rem">{badges}</div>', unsafe_allow_html=True)


def render_client_card(client: pd.Series, priority_reason: object = None, priority_maximum: int = 1) -> None:
    gid = escape(str(client.get("gid", "—")))
    evidence = client.get("evidence")
    evidence_text = str(evidence) if pd.notna(evidence) and str(evidence).strip() else "Объяснение пока не передано аналитическим пайплайном."
    reason_text = str(priority_reason) if pd.notna(priority_reason) and str(priority_reason).strip() else "Отдельное обоснование приоритета не передано. Доступное объяснение роли показано выше."
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-head"><div><div class="panel-kicker">Выбранный клиент</div><div class="subject-id">GID {gid}</div></div>{role_badge(client.get("role"))}</div>
          <div class="panel-body">
            <div class="metric-grid">
              <div class="metric"><div class="metric-label">Оценка роли</div><div class="metric-value">{format_score(client.get("role_score"))}</div></div>
              <div class="metric"><div class="metric-label">Приоритет / {priority_maximum}</div><div class="metric-value">{format_score(client.get("priority_score"), 3 if priority_maximum == 1 else 2)}</div></div>
              <div class="metric"><div class="metric-label">Кластер</div><div class="metric-value">#{escape(normalise_id(client.get("cluster_id")) or "—")}</div></div>
            </div>
            <div class="evidence"><div class="evidence-label">Обоснование роли</div>{escape(evidence_text)}</div>
            <div class="evidence"><div class="evidence-label">Причина приоритета</div>{escape(reason_text)}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_cluster(client: pd.Series, nodes: pd.DataFrame, clusters: pd.DataFrame) -> None:
    cluster_key = normalise_id(client.get("cluster_id"))
    cluster_row = clusters.loc[clusters["cluster_id"].map(normalise_id) == cluster_key] if "cluster_id" in clusters else pd.DataFrame()
    members = nodes.loc[nodes["cluster_id"].map(normalise_id) == cluster_key] if "cluster_id" in nodes and cluster_key else pd.DataFrame()
    details = cluster_row.iloc[0] if not cluster_row.empty else pd.Series(dtype=object)
    hypothesis = details.get("hypothesis", "Гипотеза кластера ещё не сформирована.")
    st.markdown(
        f"""
        <div class="panel" style="margin-top:.8rem">
          <div class="panel-head"><div><div class="panel-kicker">Контекст сети</div><div class="panel-title">Кластер #{escape(cluster_key or "—")}</div></div><span class="quiet">{len(members)} участников</span></div>
          <div class="panel-body">
            <div class="cluster-summary">
              <div class="cluster-chip"><span>Участники</span><b>{escape(str(details.get("n_nodes", len(members))))}</b></div>
              <div class="cluster-chip"><span>Seed-клиенты</span><b>{escape(str(details.get("n_seed", "—")))}</b></div>
              <div class="cluster-chip"><span>Внутренний поток</span><b>{format_money(details.get("sum_kzt_internal"))}</b></div>
            </div>
            <div class="cluster-hypothesis">{escape(str(hypothesis))}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not members.empty:
        member_text = ", ".join(escape(normalise_id(gid)) for gid in members["gid"].head(8))
        st.caption(f"Участники: {member_text}{' …' if len(members) > 8 else ''}")
        view_columns = [column for column in ("gid", "role", "role_score", "priority_score") if column in members]
        with st.expander(f"Открыть участников кластера · {len(members)}"):
            st.dataframe(
                members[view_columns].rename(
                    columns={
                        "gid": "GID",
                        "role": "Роль",
                        "role_score": "Оценка роли",
                        "priority_score": "Приоритет",
                    }
                ),
                hide_index=True,
                width="stretch",
                height=min(360, 72 + len(members) * 35),
            )


def render_top_nodes(top_nodes: pd.DataFrame, limit: int = 20, priority_maximum: int = 1, *, key: str = "top") -> None:
    if not top_nodes.empty:
        picker, action = st.columns([3, 1], vertical_alignment="bottom")
        choices = top_nodes.gid.map(normalise_id).tolist()
        labels = {normalise_id(row.gid): f"#{row.rank} · {normalise_id(row.gid)} · {role_label(row.role)}"
                  for row in top_nodes.itertuples()}
        with picker:
            gid = st.selectbox("Открыть из очереди проверки", choices, format_func=labels.get, key=f"{key}_gid")
        with action:
            st.button("Открыть GID", key=f"{key}_open", on_click=open_selected_client, args=(f"{key}_gid",), width="stretch")
    rows: list[str] = []
    for fallback_rank, (_, row) in enumerate(top_nodes.head(limit).iterrows(), start=1):
        rank = escape(str(row.get("rank", fallback_rank)))
        gid = escape(normalise_id(row.get("gid")) or "—")
        why = escape(str(row.get("why", row.get("evidence", "—"))))
        rows.append(
            f"<tr><td class='rank'>#{rank}</td><td><b>{gid}</b></td><td>{role_badge(row.get('role'))}</td><td><b>{format_score(row.get('priority_score'), 3 if priority_maximum == 1 else 2)}</b></td><td class='quiet'>{why}</td></tr>"
        )
    body = "".join(rows) or "<tr><td colspan='5' class='quiet'>Нет узлов в очереди проверки.</td></tr>"
    st.markdown(
        f"""
        <div class="panel" style="margin-top:1rem">
          <div class="panel-head"><div><div class="panel-kicker">Очередь приоритетной проверки</div><div class="panel-title">Top Nodes · первые {limit}</div></div><span class="quiet">Всего: {len(top_nodes)}</span></div>
          <div class="panel-body"><div class="table-wrap"><table class="trace-table"><thead><tr><th>Место</th><th>Клиент</th><th>Роль</th><th>Приоритет / {priority_maximum}</th><th>Причина приоритета</th></tr></thead><tbody>{body}</tbody></table></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_waiting_state(message: str) -> None:
    st.markdown(
        f"""
        <div class="panel" style="max-width:820px;margin-top:1rem">
          <div class="panel-head"><div><div class="panel-kicker">Статус пайплайна</div><div class="panel-title">Ожидание аналитических результатов</div></div><span class="quiet">Данные не загружены</span></div>
          <div class="panel-body"><div class="empty-state">{escape(message)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
