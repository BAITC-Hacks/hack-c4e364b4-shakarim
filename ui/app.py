"""TraceFlow — a read-only Streamlit workspace for financial-network investigations.

Run from the repository root:
    python -m streamlit run ui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI_DIR = Path(__file__).resolve().parent
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

import pandas as pd
import streamlit as st

from components import (
    inject_styles,
    investigation_brief,
    normalise_id,
    open_client,
    return_to_previous_client,
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
from data_access import DATA_DIR, RESULTS_DIR, read_source, raw_profiles
from insights import render_observed, render_connections_table, render_timeline, render_catalog


NODE_COLUMNS = {"gid", "role", "role_score", "priority_score", "cluster_id", "evidence"}
TOP_NODE_COLUMNS = {"rank", "gid", "role", "priority_score", "why"}
CLUSTER_COLUMNS = {"cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"}


MOCK_DIR = ROOT / "mock"


def demo_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load explicitly synthetic CSV fixtures using the production export contract."""
    tables = []
    for filename in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv", "edges.csv"):
        table, error = read_result(filename, MOCK_DIR)
        if error:
            raise ValueError(error)
        tables.append(table)
    return tuple(tables)


@st.cache_data(ttl=20, show_spinner=False)
def read_result(filename: str, directory: Path | None = None) -> tuple[pd.DataFrame, str | None]:
    """Read a ready export with a short cache; no scores are calculated here."""
    path = (directory if directory is not None else RESULTS_DIR) / filename
    if not path.exists():
        return pd.DataFrame(), f"Не найден {filename}"
    try:
        return pd.read_csv(path, encoding="utf-8-sig", dtype={"gid": "string", "cluster_id": "string", "top_gids": "string", "src": "string", "dst": "string"}), None
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
        if nodes[column].isna().any() or nodes[column].astype(str).str.strip().eq("").any():
            issues.append(f"Есть пустые значения в {column}")
    for column, maximum in (("role_score", 1), ("priority_score", 100)):
        values = pd.to_numeric(nodes[column], errors="coerce")
        if not values.between(0, maximum).all():
            issues.append(f"{column}: требуются числа от 0 до {maximum}")
    return issues


def validate_optional(table: pd.DataFrame, required: set[str], label: str) -> list[str]:
    """Keep optional exports from breaking the workspace when contracts are partial."""
    if table.empty:
        return []
    missing = sorted(required - set(table.columns))
    if missing:
        return [f"{label}: отсутствуют колонки {', '.join(missing)}"]
    blank = [column for column in required if table[column].isna().any() or table[column].astype(str).str.strip().eq("").any()]
    return [f"{label}: пустые значения в {', '.join(sorted(blank))}"] if blank else []


def prepare_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    """Add an in-memory lookup key only; all analytical values remain pipeline-owned."""
    prepared = nodes.copy()
    if "gid" in prepared:
        # The actual displayed column must be text too, not just the lookup key:
        # Arrow-backed tables and dropdowns also cross the browser boundary.
        prepared["gid"] = prepared["gid"].map(normalise_id).astype("string")
        prepared["_gid_key"] = prepared["gid"]
    flags = {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
    for column in ("is_seed", "truncated_by_depth"):
        if column in prepared:
            prepared[column] = prepared[column].map(lambda value: flags.get(str(value).strip().lower(), pd.NA)).astype("boolean")
    for column in ("role_score", "priority_score"):
        if column in prepared:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


def attach_collection_metadata(nodes: pd.DataFrame, source_nodes: pd.DataFrame,
                               directory: Path) -> tuple[pd.DataFrame, list[str]]:
    """Join upstream collection flags only; never import scores from metrics."""
    issues = []
    metadata = pd.DataFrame()
    metric_path = directory / "node_metrics.csv"
    if metric_path.exists():
        metrics, error = read_result("node_metrics.csv", directory)
        if error or "gid" not in metrics:
            issues.append(error or "node_metrics.csv: отсутствует gid; метаданные недоступны.")
        else:
            keys = metrics.gid.map(normalise_id)
            if keys.eq("").any() or keys.duplicated().any():
                issues.append("node_metrics.csv: пустые или дублирующиеся gid; метаданные не подключены.")
            else:
                columns = [c for c in ("gid", "depth", "is_seed", "truncated_by_depth") if c in metrics]
                metadata = prepare_nodes(metrics[columns]).set_index("_gid_key")
                if set(keys) != set(nodes._gid_key):
                    issues.append("node_metrics.csv: состав GID отличается от nodes_roles.csv; используются только совпавшие GID.")
    if not source_nodes.empty and "gid" in source_nodes:
        source = prepare_nodes(source_nodes)
        if source._gid_key.duplicated().any() or source._gid_key.eq("").any():
            issues.append("nodes.parquet: пустые или дублирующиеся gid; метаданные не подключены.")
        else:
            fallback = source.set_index("_gid_key")[[c for c in ("depth", "is_seed") if c in source]]
            metadata = fallback if metadata.empty else metadata.combine_first(fallback)
    result = nodes.copy()
    for column in ("depth", "is_seed", "truncated_by_depth"):
        if column in metadata and column not in result:
            result[column] = result._gid_key.map(metadata[column])
    return result, issues


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
    with st.sidebar:
        render_brand()
        source_mode = st.radio("Источник данных", ["Авто: output → mock", "Демо: mock CSV", "Исходные данные"], key="traceflow_source")
        if st.button("Обновить данные", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        with st.expander("Подключение результатов"):
            st.caption("Положите три CSV команды в эту папку и нажмите «Обновить данные». В режиме «Авто» они заменят mock.")
            st.code(str(RESULTS_DIR), language=None)
            st.caption("Другие папки задаются через TRACEFLOW_RESULTS_DIR и TRACEFLOW_DATA_DIR перед запуском.")

    # Only a genuinely absent pipeline falls back to mock. Partial, empty or
    # invalid output must remain visible as an integration issue.
    has_output = any((RESULTS_DIR / filename).exists() for filename in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv"))
    demo_mode = source_mode == "Демо: mock CSV" or (source_mode == "Авто: output → mock" and not has_output)
    raw_mode = source_mode == "Исходные данные"
    active_dir = MOCK_DIR if demo_mode else RESULTS_DIR
    pipeline_nodes, nodes_error = read_result("nodes_roles.csv", active_dir)
    pipeline_clusters, clusters_error = read_result("clusters.csv", active_dir)
    pipeline_top, top_error = read_result("top_nodes.csv", active_dir)
    pipeline_nodes = prepare_nodes(pipeline_nodes)
    validation_issues = validate_nodes(pipeline_nodes) if not pipeline_nodes.empty else []
    source_nodes, source_error = (pd.DataFrame(), None) if demo_mode else read_source("nodes")

    metadata_issues = []
    if raw_mode:
        if source_nodes.empty:
            render_pipeline_waiting_state(source_error or "Исходная таблица клиентов пуста.")
            return
        nodes = prepare_nodes(raw_profiles(source_nodes))
        clusters, top_nodes = pd.DataFrame(), pd.DataFrame()
        data_message = "Открыты исходные данные. Роли, кластеры и очередь проверки появятся после подключения результатов команды."
    elif pipeline_nodes.empty:
        waiting_message = (nodes_error or "Файл nodes_roles.csv найден, но в нём пока нет результатов для отображения.")
        render_pipeline_waiting_state(waiting_message)
        st.caption("Для независимого показа интерфейса выберите «Демо: mock CSV» в меню слева.")
        return
    elif validation_issues:
        render_pipeline_waiting_state("; ".join(validation_issues))
        with st.expander("Технические детали", expanded=False):
            st.code("\n".join(validation_issues))
        return
    else:
        nodes, clusters, top_nodes = pipeline_nodes, pipeline_clusters, pipeline_top
        if not demo_mode:
            nodes, metadata_issues = attach_collection_metadata(nodes, source_nodes, active_dir)
        data_message = "Синтетические mock CSV: роли, приоритеты и связи служат только для демонстрации UI. Это не выводы по реальным клиентам." if demo_mode else ""

    optional_issues = [
        *validate_optional(clusters, CLUSTER_COLUMNS, "clusters.csv"),
        *validate_optional(top_nodes, TOP_NODE_COLUMNS, "top_nodes.csv"),
    ]
    if optional_issues:
        if any(issue.startswith("clusters.csv") for issue in optional_issues):
            clusters = pd.DataFrame()
        if any(issue.startswith("top_nodes.csv") for issue in optional_issues):
            top_nodes = pd.DataFrame()
    if not top_nodes.empty:
        top_nodes = top_nodes.copy()
        top_nodes["priority_score"] = pd.to_numeric(top_nodes.priority_score, errors="coerce")
        top_nodes["rank"] = pd.to_numeric(top_nodes["rank"], errors="coerce")
        valid_top = (top_nodes.priority_score.between(0, 100) & top_nodes.gid.map(normalise_id).isin(nodes._gid_key)
                     & top_nodes["rank"].gt(0) & top_nodes["rank"].mod(1).eq(0))
        if not valid_top.all():
            optional_issues.append("top_nodes.csv: некорректные места, скоры или неизвестные gid")
        if top_nodes.gid.duplicated().any() or top_nodes["rank"].duplicated().any():
            optional_issues.append("top_nodes.csv: повторяющиеся gid или места в очереди")
        top_nodes = top_nodes.loc[valid_top].drop_duplicates("gid").sort_values("rank", kind="stable")
        top_nodes["rank"] = top_nodes["rank"].astype(int)
    # Display the analytics team's legacy /100 export unchanged. This is a
    # presentation scale, not a priority calculation or normalization.
    priority_maximum = 100 if (nodes.priority_score.gt(1).any() or
        (not top_nodes.empty and (top_nodes.priority_score.gt(1).any() or top_nodes.why.astype(str).str.contains("/100", regex=False).any()))) else 1

    edge_candidates = [MOCK_DIR / "edges.csv"] if demo_mode else ([DATA_DIR / "edges.parquet"] if raw_mode else [
        RESULTS_DIR / "edges.parquet", RESULTS_DIR / "edges.csv",
        DATA_DIR / "edges.parquet",
    ])
    edges, edge_path = load_edges(tuple(edge_candidates))

    with st.sidebar:
        st.markdown('<div class="side-label">Управление расследованием</div>', unsafe_allow_html=True)
        default_gid = top_nodes.iloc[0]["gid"] if not top_nodes.empty and "gid" in top_nodes else nodes.iloc[0]["gid"]
        dataset_key = (source_mode, demo_mode, str(active_dir))
        if st.session_state.get("traceflow_dataset") != dataset_key:
            st.session_state.traceflow_dataset = dataset_key
            st.session_state.traceflow_active_gid = normalise_id(default_gid)
            st.session_state.traceflow_quick_gid = normalise_id(default_gid)
            st.session_state.traceflow_history = []
            st.session_state.traceflow_last_gid = normalise_id(default_gid)
            st.session_state.traceflow_tab = "Dashboard"
        if st.session_state.get("traceflow_active_gid") not in set(nodes._gid_key):
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
            open_client(st.session_state.traceflow_quick_gid)

        st.selectbox(
            "Клиенты из выгрузки" if raw_mode else "Быстрый переход: TOP-20",
            options=[normalise_id(gid) for gid in quick_gids],
            key="traceflow_quick_gid",
            format_func=lambda gid: f"GID {gid}",
            on_change=select_quick_client,
        )
        navigation_slot = st.container()
        render_role_legend()
        source_label = "Демонстрационный кейс" if demo_mode else ("Исходная выгрузка" if raw_mode else "Готовые результаты пайплайна")
        st.markdown(f'<div class="side-caption">Источник: {source_label}<br>Роли и приоритеты UI не рассчитывает.</div>', unsafe_allow_html=True)

    selected = find_selected_client(
        nodes,
        requested_gid if submitted_search else "",
        st.session_state.traceflow_active_gid,
    )
    st.session_state.traceflow_active_gid = normalise_id(selected["gid"])
    current_gid = st.session_state.traceflow_active_gid
    previous_gid = st.session_state.get("traceflow_last_gid")
    if previous_gid and previous_gid != current_gid:
        history = st.session_state.get("traceflow_history", [])
        st.session_state.traceflow_history = (history + [previous_gid])[-10:]
        st.session_state.traceflow_tab = "Dashboard"
    st.session_state.traceflow_last_gid = current_gid
    if st.session_state.get("traceflow_history"):
        with navigation_slot:
            st.button("← Предыдущий GID", on_click=return_to_previous_client, width="stretch")
            st.caption(f"Возврат к {st.session_state.traceflow_history[-1]}")

    render_hero(demo_mode, len(nodes), len(edges), len(top_nodes), raw_mode=raw_mode)
    for issue in metadata_issues:
        st.warning(issue)
    if demo_mode:
        render_demo_banner(data_message)
    elif raw_mode:
        st.info(data_message)
        if nodes_error and not nodes_error.startswith("Не найден"):
            st.warning(nodes_error + ". Отображаются только исходные данные.")
    elif clusters_error or top_error or optional_issues:
        st.warning("Часть готовых выгрузок пока недоступна или не соответствует контракту: интерфейс показывает только доступные результаты.")
    contract_notes = []
    if priority_maximum == 100:
        contract_notes.append("Пайплайн передал приоритет в шкале 0–100. UI показывает его без пересчёта. Для итоговой сдачи по ТЗ аналитический экспорт должен использовать шкалу 0–1.")
    if not raw_mode and nodes.evidence.astype(str).str.len().gt(200).any():
        contract_notes.append("В CSV есть evidence длиннее 200 символов. UI показывает полный текст без изменений; перед сдачей команда аналитики должна сократить объяснения до лимита ТЗ.")
    if contract_notes:
        with st.sidebar:
            with st.expander(f"Качество выгрузки · {len(contract_notes)} замечания"):
                for note in contract_notes:
                    st.warning(note)
    if edges.attrs.get("load_error"):
        st.warning(edges.attrs["load_error"])
    if not demo_mode and not raw_mode and len(top_nodes) < 20:
        st.warning(f"В очереди {len(top_nodes)} узлов. Для сдачи по ТЗ требуется не менее 20.")
    if not demo_mode and not raw_mode and not source_nodes.empty:
        source_keys = set(source_nodes.gid.map(normalise_id))
        result_keys = set(nodes._gid_key)
        if source_keys != result_keys:
            st.warning(f"Состав клиентов не совпадает с исходной выгрузкой: отсутствуют {len(source_keys - result_keys)}, лишних {len(result_keys - source_keys)}. Проверьте, что подключены результаты того же кейса.")

    investigation, priorities, catalog, exports = st.tabs(["Dashboard", "Top Nodes", "Клиенты и кластеры", "Выгрузки и ограничения"], key="traceflow_tab", on_change="rerun")
    with investigation:
        st.caption("1 · Найдите GID   →   2 · Проверьте роль и причину приоритета   →   3 · Проследите входящие и исходящие связи")
        profile_column, graph_column = st.columns([1.03, 1.97], gap="large")
        with profile_column:
            reason = selected.get("priority_reason", selected.get("priority_evidence"))
            if not top_nodes.empty:
                priority_row = top_nodes.loc[top_nodes.gid.map(normalise_id).eq(normalise_id(selected.gid))]
                if not priority_row.empty:
                    reason = priority_row.iloc[0]["why"]
            render_client_card(selected, priority_reason=reason, priority_maximum=priority_maximum)
            st.caption("Роль — гипотеза для проверки. Оценка роли не является вероятностью нарушения.")
            st.download_button("Скачать сводку по GID", investigation_brief(selected, reason, edges, source_label, priority_maximum).encode("utf-8"),
                               file_name=f"traceflow_{current_gid}.md", mime="text/markdown", width="stretch")
            if not edges.attrs.get("load_error"):
                render_observed(selected, edges)
            if not raw_mode:
                render_cluster(selected, nodes, clusters)
        with graph_column:
            render_client_connections(selected["gid"], edges, nodes, demo_mode=demo_mode)
        if not raw_mode:
            render_top_nodes(top_nodes, limit=5, priority_maximum=priority_maximum, key="dashboard_top")
        with st.expander("Связи и история переводов", expanded=False):
            if not edges.attrs.get("load_error"):
                render_connections_table(selected["gid"], edges, nodes)
            if not demo_mode:
                transactions, tx_error = read_source("transactions")
                render_timeline(selected["gid"], transactions, tx_error)
    with priorities:
        render_top_nodes(top_nodes, priority_maximum=priority_maximum, key="all_top")
        st.caption("В списке перехода доступны все узлы очереди. Кнопка «Открыть GID» возвращает на Dashboard. Порядок rank, приоритет и объяснение переданы аналитикой.")
    with catalog:
        render_catalog(nodes, raw_mode)
        if not clusters.empty:
            st.markdown("#### Кластеры сети")
            st.dataframe(clusters, hide_index=True, width="stretch")
    with exports:
        st.markdown("#### Результаты для проверки")
        if raw_mode:
            st.info("Аналитические выгрузки пока не переданы. Экспорт ролей станет доступен после их подключения.")
        elif demo_mode:
            st.info("Это вымышленный дизайн-кейс. Он не является результатом анализа датасета.")
        else:
            st.caption("Скачиваются исходные CSV команды без изменений. Предупреждения о контракте нужно устранить в пайплайне.")
            for filename in ("nodes_roles.csv", "clusters.csv", "top_nodes.csv"):
                path = RESULTS_DIR / filename
                if path.exists():
                    st.download_button(f"Скачать {filename}", path.read_bytes(), file_name=filename, mime="text/csv")
            if optional_issues:
                st.warning("; ".join(optional_issues))
        st.markdown("#### Границы наблюдения")
        st.markdown("Июль 2026 · внутрибанковские переводы от 5 000 ₸ · обход только по исходящим, до 4 колен.\n\n"
                    "На границе обхода дальнейшие переводы неизвестны. Входящий поток seed-клиентов неполон. "
                    "В графе есть клиенты без наблюдаемых связей. Роли и кластеры — гипотезы для проверки, а не вывод о виновности.")
    if edge_path and not demo_mode:
        st.caption(f"Источник связей: {edge_path.name}")


if __name__ == "__main__":
    main()
