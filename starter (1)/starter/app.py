"""Streamlit explorer for generated graph analysis CSV files."""
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st
from pyvis.network import Network
import streamlit.components.v1 as components

st.set_page_config(page_title="Money Graph Explorer", page_icon="🔎", layout="wide")
st.title("🔎 Money Graph Explorer")
st.caption("Обзор сети переводов, ролей и узлов для проверки")

DATA_DIR = Path(__file__).parent / "output"

@st.cache_data
def read_csvs():
    def read(name):
        p = DATA_DIR / name
        return pd.read_csv(p) if p.exists() else pd.DataFrame()
    return read("node_metrics.csv"), read("nodes_roles.csv"), read("clusters.csv"), read("top_nodes.csv")

metrics, roles, clusters, top = read_csvs()
if metrics.empty or roles.empty:
    st.warning("CSV пока нет. Сначала выполните: `python starter.py --data ../data --out ./output`.")
    st.stop()

tabs = st.tabs(["Dashboard", "Graph", "Top Nodes"])
with tabs[0]:
    a,b,c,d=st.columns(4)
    a.metric("Узлов",f"{len(metrics):,}")
    aedges=metrics.out_degree.sum()
    b.metric("Связей",f"{int(aedges):,}")
    tx=int(metrics.out_tx_count.sum()) if "out_tx_count" in metrics else 0
    c.metric("Транзакций",f"{tx:,}")
    d.metric("Кластеров",f"{len(clusters):,}")
    st.subheader("Распределение ролей")
    st.bar_chart(roles.role.value_counts())
    st.subheader("Крупнейшие кластеры")
    if not clusters.empty: st.dataframe(clusters.sort_values("n_nodes",ascending=False), width="stretch", hide_index=True)

with tabs[1]:
    gid_text=st.text_input("Поиск по GID",placeholder="Например, 582")
    if gid_text:
        try:
            gid=int(gid_text)
            row=roles.loc[roles.gid==gid]
            if row.empty: st.error(f"GID {gid} не найден")
            else:
                r=row.iloc[0]; m=metrics.loc[metrics.gid==gid].iloc[0]
                st.subheader(f"GID {gid} · {r.role}")
                x,y,z=st.columns(3); x.metric("Priority",f"{r.priority_score:.3f}"); y.metric("Role score",f"{r.role_score:.3f}"); z.metric("Cluster",str(r.cluster_id))
                st.write(r.evidence)
                # The metric export contains node measures; the original edge list is optional for graph rendering.
                edge_file=Path(__file__).resolve().parents[2] / "data (1)" / "data" / "edges.parquet"
                if edge_file.exists():
                    e=pd.read_parquet(edge_file); near=e[(e.src==gid)|(e.dst==gid)].head(100)
                    if not near.empty:
                        net=Network(height="520px",width="100%",directed=True,bgcolor="#0e1117",font_color="white")
                        neighbors=set(near.src)|set(near.dst)
                        lookup=roles.set_index("gid")
                        for n in neighbors:
                            rr=lookup.loc[n] if n in lookup.index else None
                            title=f"GID {n}" if rr is None else f"GID {n} · {rr.role} · priority {rr.priority_score:.2f}"
                            net.add_node(int(n),label=str(n),title=title,color="#f59e0b" if n==gid else "#38bdf8",size=28 if n==gid else 18)
                        for edge in near.itertuples(index=False): net.add_edge(int(edge.src),int(edge.dst),value=max(1,float(edge.sum_kzt)**.25),title=f"{edge.sum_kzt:,.0f} KZT · {edge.n_tx} tx")
                        components.html(net.generate_html(),height=540,scrolling=True)
                    else: st.info("Для узла нет ребер в исходной выгрузке.")
                else: st.info("Файл edges.parquet недоступен в ожидаемой папке. Метрики и связи доступны в CSV.")
        except ValueError: st.error("Введите числовой GID.")
    else: st.info("Введите GID, чтобы посмотреть роль, обоснование и соседние переводы.")

with tabs[2]:
    if not top.empty: st.dataframe(top, width="stretch", hide_index=True)
    else: st.info("top_nodes.csv пока пуст.")
