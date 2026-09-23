"""Observed evidence only: flows, dates and drill-down, never role inference."""
import pandas as pd
import streamlit as st

from components import format_money, normalise_id, role_label


def incident_edges(gid: object, edges: pd.DataFrame) -> pd.DataFrame:
    key = normalise_id(gid)
    src = edges.get("_src_key", edges.src.map(normalise_id))
    dst = edges.get("_dst_key", edges.dst.map(normalise_id))
    return edges.loc[src.eq(key) | dst.eq(key)].copy()


def render_collection_caveats(client: pd.Series) -> None:
    """Collection limits still apply when an edge export cannot be loaded."""
    depth = client.get("depth")
    boundary_flag = str(client.get("truncated_by_depth", "")).lower() in {"true", "1", "1.0"}
    if boundary_flag or (pd.notna(depth) and str(depth) in {"4", "4.0"}):
        st.warning("Граница 4-го колена: дальнейшие переводы могут не наблюдаться. Это не доказательство того, что деньги остались у клиента.")
    if str(client.get("is_seed", "")).lower() in {"true", "1", "1.0"}:
        st.info("Исходный клиент (seed). Его входящий поток неполон; отношение отправленного к полученному не отражает полный баланс.")


def render_observed(client: pd.Series, edges: pd.DataFrame) -> None:
    key = normalise_id(client.gid)
    incident = incident_edges(key, edges)
    incoming = incident.loc[incident.dst.map(normalise_id).eq(key)]
    outgoing = incident.loc[incident.src.map(normalise_id).eq(key)]
    cols = st.columns(2)
    for col, label, table, peer in [(cols[0], "Получено в графе", incoming, "src"),
                                  (cols[1], "Отправлено в графе", outgoing, "dst")]:
        col.metric(label, format_money(table.sum_kzt.sum()) if "sum_kzt" in table else "—")
        col.caption(f"Контрагентов: {table[peer].nunique()}")
    st.caption("Суммы относятся только к наблюдаемой сети. Роли — гипотезы для проверки.")


def render_connections_table(gid: object, edges: pd.DataFrame, nodes: pd.DataFrame) -> None:
    table = incident_edges(gid, edges)
    st.markdown("#### Все связи клиента")
    if table.empty:
        st.info("В выгрузке нет переводов этого клиента. Отсутствие связей не означает отсутствие активности за пределами выборки.")
        return
    if "sum_kzt" in table:
        table = table.sort_values("sum_kzt", ascending=False, kind="stable")
    key = normalise_id(gid)
    table["Направление"] = table.src.map(normalise_id).map(lambda src: "Исходящий →" if src == key else "← Входящий")
    table["Контрагент"] = [normalise_id(row.dst if normalise_id(row.src) == key else row.src) for row in table.itertuples()]
    fields = [c for c in ["Направление", "Контрагент", "src", "dst", "sum_kzt", "n_tx"] if c in table]
    st.dataframe(table[fields].rename(columns={"src": "Отправитель", "dst": "Получатель", "sum_kzt": "Сумма, ₸", "n_tx": "Переводов"}),
                 hide_index=True, width="stretch", height=min(350, 38 + 35 * len(table)))
    available = set(nodes.gid.map(normalise_id))
    peers = [peer for peer in table["Контрагент"].unique() if peer in available and peer != key]
    if peers:
        choice = st.selectbox("Перейти к связанному клиенту", peers, key=f"peer_{key}")
        if st.button("Открыть контрагента", key=f"open_peer_{key}"):
            st.session_state.traceflow_active_gid = choice
            st.rerun()
    st.download_button("Скачать все связи CSV", table[fields].to_csv(index=False).encode("utf-8-sig"),
                       file_name=f"connections_{key}.csv", mime="text/csv")


def render_timeline(gid: object, transactions: pd.DataFrame, error: str | None) -> None:
    st.markdown("#### Динамика переводов")
    if error:
        st.info(error)
        return
    key = normalise_id(gid)
    if transactions.empty:
        st.info("Нет транзакций для построения динамики.")
        return
    incoming = transactions.loc[transactions.dst.map(normalise_id).eq(key)]
    outgoing = transactions.loc[transactions.src.map(normalise_id).eq(key)]
    if incoming.empty and outgoing.empty:
        st.info("У клиента нет отдельных транзакций в этой выгрузке.")
        return
    metric = st.radio("Показатель динамики", ["Сумма, ₸", "Количество переводов"], horizontal=True)
    def aggregate(frame):
        grouped = frame.groupby(frame.date.dt.normalize()).sum_kzt
        return grouped.sum() if metric == "Сумма, ₸" else grouped.size()
    daily = pd.concat([aggregate(incoming).rename("Входящие"), aggregate(outgoing).rename("Исходящие")], axis=1, sort=True).fillna(0)
    dates = pd.date_range(transactions.date.min().normalize(), transactions.date.max().normalize())
    daily = daily.reindex(dates, fill_value=0)
    daily.index.name = "Дата"
    st.bar_chart(daily, color=["#47B4FF", "#FF9F5A"], stack=False, height=260)
    st.caption("Доступны даты без времени суток. Совпадение входа и выхода в один день само по себе не доказывает сквозной транзит.")


def render_catalog(nodes: pd.DataFrame, raw_mode: bool) -> None:
    st.markdown("#### Каталог клиентов")
    roles = sorted(nodes.role.unique())
    selected_roles = st.multiselect("Роль", roles, format_func=role_label)
    clusters = sorted(nodes.cluster_id.map(normalise_id).unique())
    selected_clusters = st.multiselect("Кластер", clusters, disabled=raw_mode)
    view = nodes
    if selected_roles:
        view = view.loc[view.role.isin(selected_roles)]
    if selected_clusters:
        view = view.loc[view.cluster_id.map(normalise_id).isin(selected_clusters)]
    if not raw_mode:
        view = view.sort_values("priority_score", ascending=False, kind="stable")
    cols = [c for c in ["gid", "role", "priority_score", "cluster_id", "depth", "is_seed", "evidence"] if c in view]
    st.caption(f"Найдено {len(view):,} из {len(nodes):,} клиентов".replace(",", " "))
    st.dataframe(view[cols], hide_index=True, width="stretch", height=360)
    if not view.empty:
        choice = st.selectbox("Открыть клиента из каталога", view.gid.map(normalise_id).tolist())
        if st.button("Открыть выбранного клиента"):
            st.session_state.traceflow_active_gid = choice
            st.rerun()
