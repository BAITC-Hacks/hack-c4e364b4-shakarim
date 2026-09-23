"""Network graph metrics calculation for transaction nodes."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx


def compute_node_metrics(G: nx.DiGraph, nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """One row per gid; `truncated_by_depth` flags censored leaves explicitly."""
    node = nodes[["gid", "depth", "is_seed"]].drop_duplicates("gid").copy()
    node["is_seed"] = node.is_seed.astype(bool)

    # Edge table is already aggregated by ordered pair; distinct pair count is degree.
    f = edges.groupby("src").agg(
        out_amount=("sum_kzt", "sum"),
        out_tx_count=("n_tx", "sum"),
        unique_receivers=("dst", "nunique")
    )
    f = f.join(
        edges.groupby("dst").agg(
            in_amount=("sum_kzt", "sum"),
            in_tx_count=("n_tx", "sum"),
            unique_senders=("src", "nunique")
        ),
        how="outer"
    )
    deg = pd.DataFrame({
        "gid": list(G),
        "in_degree": [G.in_degree(x) for x in G],
        "out_degree": [G.out_degree(x) for x in G]
    })
    f = node.merge(deg, on="gid", how="left").join(f, on="gid").fillna(0)
    for c in ("in_degree", "out_degree", "in_tx_count", "out_tx_count", "unique_senders", "unique_receivers"):
        f[c] = f[c].astype(int)

    # At depth four, absent outflow is censored by extraction, not evidence of a true sink.
    f["truncated_by_depth"] = (f.depth >= 4) & (f.out_degree == 0)
    f["pass_through"] = np.divide(
        f.out_amount,
        f.in_amount,
        out=np.full(len(f), np.nan),
        where=f.in_amount.to_numpy() > 0
    )
    # Suppress misleading ratios on seeds: their incoming history lies outside the extract.
    f.loc[f.is_seed, "pass_through"] = np.nan

    pr = nx.pagerank(G, weight="sum_kzt") if G.number_of_nodes() else {}
    f["pagerank"] = f.gid.map(pr).fillna(0.0)

    # Weighted betweenness uses inverse transfer volume as path cost.
    for u, v, d in G.edges(data=True):
        d["distance"] = 1.0 / max(d.get("sum_kzt", 0.0), 1e-12)
    btw = (
        nx.betweenness_centrality(G, weight="distance", normalized=True)
        if len(G) < 10000
        else nx.betweenness_centrality(G, k=min(500, len(G)), weight="distance", seed=42)
    )
    f["betweenness"] = f.gid.map(btw).fillna(0.0)
    return f


def save_node_metrics(df: pd.DataFrame, output_path: Path | str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
