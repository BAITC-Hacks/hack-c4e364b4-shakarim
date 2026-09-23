"""TraceFlow — a read-only Streamlit workspace for financial-network investigations.

Run from the repository root:
    python -m streamlit run ui/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from components import (
    inject_styles,
    normalise_id,
    render_brand,
    render_client_card,
    render_cluster,
    render_demo_banner,
    render_hero,
    render_pipeline_waiting_state,
    render_role_legend,
    render_top_nodes,
)
from graph_view import load_edges, render_client_connections


ROOT = Path(__file__).resolve().parents[1]
# After analytics integration, change only this line to ROOT / "output".
RESULTS_DIR = ROOT / "mock"
NODE_COLUMNS = {"gid", "role", "role_score", "priority_score", "cluster_id", "evidence"}
TOP_NODE_COLUMNS = {"gid", "priority_score"}
CLUSTER_COLUMNS = {"cluster_id"}


def demo_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Explicit presentation-only preview; never mixed with pipeline results."""
    nodes = pd.DataFrame(
        [
            [901245, "coordinator", 0.94, 0.98, 17, "Связывает 3 ветки; через узел проходит ключевой маршрут между кластером сбора и распределения."],
            [902188, "consolidator", 0.89, 0.92, 17, "Получает средства от 8 разных участников и концентрирует поток перед передачей дальше."],
            [903470, "transit", 0.86, 0.87, 17, "Передаёт большую часть наблюдаемого входящего потока в следующую ветку."],
            [904311, "distributor", 0.91, 0.90, 17, "Формирует веерный вывод на 6 получателей внутри наблюдаемого маршрута."],
            [905624, "terminal", 0.78, 0.72, 17, "Получает средства и не имеет наблюдаемых исходящих связей в доступной части сети."],
            [906810, "peripheral", 0.41, 0.38, 21, "Имеет ограниченное число связей и не формирует выраженную структурную роль."],
            [907193, "transit", 0.73, 0.70, 21, "Поддерживает короткий транзитный маршрут между двумя участниками."],
        ],
        columns=["gid", "role", "role_score", "priority_score", "cluster_id", "evidence"],
    )
    clusters = pd.DataFrame(
        [
            [17, 5, 2, 12_480_000, "901245, 902188, 903470, 904311, 905624", "Вероятная цепочка консолидации и веерного распределения средств."],
            [21, 2, 0, 1_230_000, "906810, 907193", "Небольшая периферийная транзитная ветка."],
        ],
        columns=["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"],
    )
    top_nodes = pd.DataFrame(
        [
            [1, 901245, "coordinator", 0.98, "Соединяет 3 ветки и контролирует критическую точку маршрута."],
            [2, 902188, "consolidator", 0.92, "Концентрирует средства от 8 участников."],
            [3, 904311, "distributor", 0.90, "Распределяет поток на 6 получателей."],
            [4, 903470, "transit", 0.87, "Передаёт основной наблюдаемый поток дальше."],
            [5, 905624, "terminal", 0.72, "Конечная точка наблюдаемой ветки."],
            [6, 907193, "transit", 0.70, "Поддерживает вспомогательную транзитную цепочку."],
            [7, 906810, "peripheral", 0.38, "Периферийная ветка без выраженного влияния."],
        ],
        columns=["rank", "gid", "role", "priority_score", "why"],
    )
    edges = pd.DataFrame(
        [
            [902188, 901245, 4_840_000], [903470, 901245, 2_100_000], [901245, 904311, 5_930_000],
            [901245, 905624, 1_420_000], [904311, 905624, 2_360_000], [904311, 903470, 1_100_000],
        ],
        columns=["src", "dst", "sum_kzt"],
    )
    return nodes, clusters, top_nodes, edges


@st.cache_data(ttl=20, show_spinner=False)
def read_result(filename: str) -> tuple[pd.DataFrame, str | None]:
    """Read a ready export with a short cache; no scores are calculated here."""
    path = RESULTS_DIR / filename
    if not path.exists():
        return pd.DataFrame(), f"Не найден {filename}"
    try:
        return pd.read_csv(path, encoding="utf-8-sig"), None
    except Exception:
        return pd.DataFrame(), f"Не удалось прочитать {filename}"


def validate_nodes(nodes: pd.DataFrame) -> list[str]:
    missing = sorted(NODE_COLUMNS - set(nodes.columns))
    if missing:
        return [f"Отсутствуют колонки: {', '.join(missing)}"]
    keys = nodes["_gid_key"] if "_gid_key" in nodes else nodes["gid"].map(normalise_id)
    issues: list[str] = []
    if keys.eq("").any():
        issues.append("Есть строки с пустым gid")
    if keys[keys.ne("")].duplicated().any():
        issues.append("Есть дублирующиеся gid")
    for column in NODE_COLUMNS - {"gid"}:
        if nodes[column].isna().any():
            issues.append(f"Есть пустые значения в {column}")
    return issues


def validate_optional(table: pd.DataFrame, required: set[str], label: str) -> list[str]:
    """Keep optional exports from breaking the workspace when contracts are partial."""
    if table.empty:
        return []
    missing = sorted(required - set(table.columns))
    return [f"{label}: отсутствуют колонки {', '.join(missing)}"] if missing else []


def prepare_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    """Add an in-memory lookup key only; all analytical values remain pipeline-owned."""
    prepared = nodes.copy()
    if "gid" in prepared:
        prepared["_gid_key"] = prepared["gid"].map(normalise_id)
    return prepared


def find_selected_client(nodes: pd.DataFrame, requested_gid: str, fallback_gid: object) -> pd.Series:
    keys = nodes["_gid_key"] if "_gid_key" in nodes else nodes["gid"].map(normalise_id)
    if requested_gid.strip():
        matched = nodes.loc[keys == normalise_id(requested_gid)]
        if not matched.empty:
            return matched.iloc[0]
        st.sidebar.warning("Клиент не найден. Показан выбранный профиль.")
    matched = nodes.loc[keys == normalise_id(fallback_gid)]
    return matched.iloc[0] if not matched.empty else nodes.iloc[0]


def main() -> None:
    st.set_page_config(page_title="TraceFlow | Financial Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    inject_styles()

    pipeline_nodes, nodes_error = read_result("nodes_roles.csv")
    pipeline_clusters, clusters_error = read_result("clusters.csv")
    pipeline_top, top_error = read_result("top_nodes.csv")
    pipeline_nodes = prepare_nodes(pipeline_nodes)
    validation_issues = validate_nodes(pipeline_nodes) if not pipeline_nodes.empty else []
    # A design preview is opt-in. An existing-but-empty export must never be
    # presented as an investigation result or silently replaced with test data.
    preview_enabled = bool(st.session_state.get("traceflow_preview_enabled", False))
    demo_mode = pipeline_nodes.empty and nodes_error is not None and preview_enabled

    if demo_mode:
        nodes, clusters, top_nodes, demo_edges = demo_results()
        nodes = prepare_nodes(nodes)
        data_message = "Пайплайн ещё не передал CSV. На экране показан демонстрационный кейс — это не аналитический вывод."
    elif pipeline_nodes.empty:
        with st.sidebar:
            render_brand()
            st.markdown('<div class="side-label">Предпросмотр интерфейса</div>', unsafe_allow_html=True)
            st.caption("Открывает только маркированный дизайн-кейс. Он не подменяет результаты пайплайна.")
            if st.button("Открыть дизайн-кейс", width="stretch"):
                st.session_state.traceflow_preview_enabled = True
                st.rerun()
        waiting_message = (
            "Готовая выгрузка nodes_roles.csv пока недоступна. После появления экспорта интерфейс автоматически покажет только результаты пайплайна."
            if nodes_error
            else "Файл nodes_roles.csv найден, но в нём пока нет результатов для отображения."
        )
        render_pipeline_waiting_state(waiting_message)
        return
    elif validation_issues:
        with st.sidebar:
            render_brand()
        render_pipeline_waiting_state("; ".join(validation_issues))
        with st.expander("Технические детали", expanded=False):
            st.code("\n".join(validation_issues))
        return
    else:
        nodes, clusters, top_nodes = pipeline_nodes, pipeline_clusters, pipeline_top
        demo_edges = pd.DataFrame(columns=["src", "dst", "sum_kzt"])
        data_message = ""

    optional_issues = [
        *validate_optional(clusters, CLUSTER_COLUMNS, "clusters.csv"),
        *validate_optional(top_nodes, TOP_NODE_COLUMNS, "top_nodes.csv"),
    ]
    if optional_issues:
        if any(issue.startswith("clusters.csv") for issue in optional_issues):
            clusters = pd.DataFrame()
        if any(issue.startswith("top_nodes.csv") for issue in optional_issues):
            top_nodes = pd.DataFrame()

    edge_candidates = [
        RESULTS_DIR / "edges.parquet", RESULTS_DIR / "edges.csv",
    ]
    loaded_edges, edge_path = load_edges(tuple(edge_candidates))
    edges = demo_edges if demo_mode else loaded_edges

    with st.sidebar:
        render_brand()
        st.markdown('<div class="side-label">Управление расследованием</div>', unsafe_allow_html=True)
        default_gid = top_nodes.iloc[0]["gid"] if not top_nodes.empty and "gid" in top_nodes else nodes.iloc[0]["gid"]
        if "traceflow_active_gid" not in st.session_state:
            st.session_state.traceflow_active_gid = normalise_id(default_gid)

        with st.form("gid-search", clear_on_submit=True):
            requested_gid = st.text_input("Найти клиента по GID", placeholder="Введите GID клиента", label_visibility="collapsed")
            submitted_search = st.form_submit_button("Открыть профиль", width="stretch")

        quick_gids = (
            top_nodes["gid"].head(20).tolist()
            if not top_nodes.empty and "gid" in top_nodes
            else nodes["gid"].head(20).tolist()
        )
        quick_keys = {normalise_id(gid) for gid in quick_gids}
        if st.session_state.get("traceflow_quick_gid") not in quick_keys:
            st.session_state.traceflow_quick_gid = normalise_id(default_gid) if normalise_id(default_gid) in quick_keys else normalise_id(quick_gids[0])

        def select_quick_client() -> None:
            st.session_state.traceflow_active_gid = st.session_state.traceflow_quick_gid

        st.selectbox(
            "Быстрый переход: TOP-20",
            options=[normalise_id(gid) for gid in quick_gids],
            key="traceflow_quick_gid",
            format_func=lambda gid: f"GID {gid}",
            on_change=select_quick_client,
        )
        if st.button("Обновить данные", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        if demo_mode and st.button("Закрыть дизайн-кейс", width="stretch"):
            st.session_state.traceflow_preview_enabled = False
            st.rerun()
        render_role_legend()
        source_label = "Демонстрационный кейс" if demo_mode else "Готовые результаты пайплайна"
        st.markdown(f'<div class="side-caption">Источник: {source_label}<br>Роли и приоритеты UI не рассчитывает.</div>', unsafe_allow_html=True)

    selected = find_selected_client(
        nodes,
        requested_gid if submitted_search else "",
        st.session_state.traceflow_active_gid,
    )
    st.session_state.traceflow_active_gid = normalise_id(selected["gid"])

    render_hero(demo_mode, len(nodes), len(edges), len(top_nodes))
    if demo_mode:
        render_demo_banner(data_message)
    elif clusters_error or top_error or optional_issues:
        st.warning("Часть готовых выгрузок пока недоступна или не соответствует контракту: интерфейс показывает только доступные результаты.")

    profile_column, graph_column = st.columns([1.03, 1.97], gap="large")
    with profile_column:
        render_client_card(selected)
        render_cluster(selected, nodes, clusters)
    with graph_column:
        render_client_connections(selected["gid"], edges, nodes, demo_mode=demo_mode)

    render_top_nodes(top_nodes)
    if edge_path and not demo_mode:
        st.caption(f"Источник связей: {edge_path.name}")


if __name__ == "__main__":
    main()
