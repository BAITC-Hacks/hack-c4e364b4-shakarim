#!/usr/bin/env python3
"""Build node metrics, behavioral roles, communities, and review priorities."""
from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

ROLES = ["consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral"]


def load(data_dir: Path):
    return tuple(pd.read_parquet(data_dir / f) for f in ("edges.parquet", "nodes.parquet", "transactions.parquet"))


def build_graph(edges: pd.DataFrame, nodes: pd.DataFrame) -> nx.DiGraph:
    """Directed, weighted graph. Keep isolated nodes from nodes.parquet."""
    G = nx.DiGraph()
    G.add_nodes_from(nodes.gid.tolist())
    for r in edges.itertuples(index=False):
        G.add_edge(r.src, r.dst, sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx), depth=int(r.depth))
    return G


def basic_features(G: nx.DiGraph, nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """One row per gid; `truncated_by_depth` flags censored leaves explicitly."""
    node = nodes[["gid", "depth", "is_seed"]].drop_duplicates("gid").copy()
    node["is_seed"] = node.is_seed.astype(bool)
    # Edge table is already aggregated by ordered pair; distinct pair count is degree.
    f = edges.groupby("src").agg(out_amount=("sum_kzt", "sum"), out_tx_count=("n_tx", "sum"), unique_receivers=("dst", "nunique"))
    f = f.join(edges.groupby("dst").agg(in_amount=("sum_kzt", "sum"), in_tx_count=("n_tx", "sum"), unique_senders=("src", "nunique")), how="outer")
    deg = pd.DataFrame({"gid": list(G), "in_degree": [G.in_degree(x) for x in G], "out_degree": [G.out_degree(x) for x in G]})
    f = node.merge(deg, on="gid", how="left").join(f, on="gid").fillna(0)
    for c in ("in_degree", "out_degree", "in_tx_count", "out_tx_count", "unique_senders", "unique_receivers"):
        f[c] = f[c].astype(int)
    # At depth four, absent outflow is censored by extraction, not evidence of a true sink.
    f["truncated_by_depth"] = (f.depth >= 4) & (f.out_degree == 0)
    f["pass_through"] = np.divide(f.out_amount, f.in_amount, out=np.full(len(f), np.nan), where=f.in_amount.to_numpy() > 0)
    # Suppress misleading ratios on seeds: their incoming history lies outside the extract.
    f.loc[f.is_seed, "pass_through"] = np.nan
    pr = nx.pagerank(G, weight="sum_kzt") if G.number_of_nodes() else {}
    f["pagerank"] = f.gid.map(pr).fillna(0.0)
    # Weighted betweenness uses inverse transfer volume as path cost.
    for u, v, d in G.edges(data=True):
        d["distance"] = 1.0 / max(d.get("sum_kzt", 0.0), 1e-12)
    btw = nx.betweenness_centrality(G, weight="distance", normalized=True) if len(G) < 10000 else nx.betweenness_centrality(G, k=min(500, len(G)), weight="distance", seed=42)
    f["betweenness"] = f.gid.map(btw).fillna(0.0)
    return f


def _norm(s: pd.Series) -> pd.Series:
    x = np.log1p(s.clip(lower=0))
    lo, hi = x.quantile(.05), x.quantile(.95)
    return ((x - lo) / (hi - lo)).clip(0, 1) if hi > lo else pd.Series(0.0, index=s.index)


def analyze(df: pd.DataFrame, G: nx.DiGraph):
    """Assign explainable heuristic roles/scores and weighted Louvain communities."""
    U = nx.Graph()
    U.add_nodes_from(G.nodes)
    for u, v, d in G.edges(data=True):
        if U.has_edge(u, v): U[u][v]["weight"] += d.get("sum_kzt", 0)
        else: U.add_edge(u, v, weight=d.get("sum_kzt", 0))
    communities = nx.community.louvain_communities(U, weight="weight", seed=42) if U.number_of_edges() else [{n} for n in U]
    cmap = {gid: i for i, comm in enumerate(sorted(communities, key=lambda c: min(map(str, c)))) for gid in comm}
    df["cluster_id"] = df.gid.map(cmap).fillna(-1).astype(int)
    # Robust within-dataset scaling; seeds have incomplete inbound amounts, so don't score that signal for them.
    for name in ("in_degree", "out_degree", "in_amount", "out_amount", "pagerank", "betweenness", "unique_senders", "unique_receivers"):
        df["_" + name] = _norm(df[name])
    ratio = df.pass_through
    roles, role_scores, evidence, priorities = [], [], [], []
    for _, r in df.iterrows():
        inbound = 0 if r.is_seed else r.in_degree
        if r.truncated_by_depth:
            role = "peripheral" if r.in_degree == 0 else "transit"
        elif inbound >= max(2, int(df.in_degree.quantile(.75))) and r.out_degree <= max(1, int(df.out_degree.median())):
            role = "consolidator"
        elif r.out_degree >= max(3, int(df.out_degree.quantile(.75))) and r.out_degree > r.in_degree:
            role = "distributor"
        elif r.in_degree > 0 and r.out_degree > 0 and pd.notna(r.pass_through) and .8 <= r.pass_through <= 1.2:
            role = "transit"
        elif r.in_degree > 0 and r.out_degree == 0:
            role = "terminal"
        elif r.betweenness >= df.betweenness.quantile(.9) and (r.in_degree > 0 and r.out_degree > 0):
            role = "coordinator"
        else:
            role = "peripheral"
        roles.append(role)
        norm = lambda n: float(r["_" + n])
        rs = {"consolidator": .55*norm("unique_senders")+.25*norm("in_amount")+.2*norm("in_degree"),
              "distributor": .4*norm("unique_receivers")+.35*norm("out_amount")+.25*norm("out_degree"),
              "transit": 1-min(1, abs(np.log(max(r.pass_through, 1e-9))) / 2) if pd.notna(r.pass_through) else .35,
              "coordinator": .65*norm("betweenness")+.35*norm("pagerank"),
              "terminal": min(1, .35+.25*norm("in_amount")+.2*norm("in_degree")), "peripheral": .35}
        role_scores.append(round(float(np.clip(rs[role], 0, 1)), 4))
        why = f"{r.in_degree} входящих связей от {r.unique_senders} отправителей; {r.out_degree} исходящих связей к {r.unique_receivers} получателям; вход {r.in_amount:,.0f} KZT, выход {r.out_amount:,.0f} KZT"
        if r.truncated_by_depth: why += ". Depth=4: отсутствие исходящих связей может быть следствием границы выгрузки"
        elif r.is_seed: why += ". Seed: входящие суммы неполны, pass-through не интерпретируется"
        evidence.append(why[:200])
    df["role"] = roles
    df["role_score"] = role_scores
    df["evidence"] = evidence
    df["priority_score"] = (0.25*df["_pagerank"] + 0.22*df["_betweenness"] + 0.18*df["_in_amount"] + 0.18*df["_out_amount"] + 0.17*df["_in_degree"].combine(df["_out_degree"], max)).clip(0, 1).round(4)
    return df, U


def write_outputs(df: pd.DataFrame, U: nx.Graph, tx: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "node_metrics.csv", index=False)
    cols = ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence", "in_deg", "out_deg", "in_kzt", "out_kzt", "pagerank", "pass_through", "depth", "is_seed", "truncated_by_depth"]
    roles = df.rename(columns={"in_degree":"in_deg", "out_degree":"out_deg", "in_amount":"in_kzt", "out_amount":"out_kzt"})
    roles[cols].to_csv(out_dir / "nodes_roles.csv", index=False)
    crows=[]
    for cid, group in df.groupby("cluster_id"):
        gids=set(group.gid); internal=sum(d.get("sum_kzt",0) for u,v,d in U.edges(data=True) if u in gids and v in gids)
        tops=group.nlargest(5,"priority_score").gid.astype(str).tolist()
        dominant=group.role.value_counts().idxmax()
        crows.append(dict(cluster_id=cid,n_nodes=len(group),n_seed=int(group.is_seed.sum()),sum_kzt_internal=internal,top_gids=";".join(tops),hypothesis=f"Преобладает роль {dominant}; гипотеза требует проверки по связям"))
    pd.DataFrame(crows, columns=["cluster_id","n_nodes","n_seed","sum_kzt_internal","top_gids","hypothesis"]).to_csv(out_dir/"clusters.csv",index=False)
    top=df.nlargest(min(100,len(df)),"priority_score").copy()
    top.insert(0,"rank",range(1,len(top)+1)); top.rename(columns={"gid":"gid"},inplace=True)
    top[["rank","gid","role","priority_score","evidence"]].rename(columns={"evidence":"why"}).to_csv(out_dir/"top_nodes.csv",index=False)
    print(f"Nodes: {len(df):,}; edges: {U.number_of_edges():,}; transactions: {int(tx.shape[0]):,}; clusters: {df.cluster_id.nunique():,}")
    print(f"Wrote node_metrics.csv, nodes_roles.csv, clusters.csv, top_nodes.csv to {out_dir}")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",type=Path,default=Path("../data")); ap.add_argument("--out",type=Path,default=Path("./output"))
    a=ap.parse_args(); edges,nodes,tx=load(a.data)
    G=build_graph(edges,nodes); df=basic_features(G,nodes,edges); df,_=analyze(df,G); write_outputs(df,nx.Graph(G.to_undirected()),tx,a.out)

if __name__=="__main__": main()
