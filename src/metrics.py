"""Network graph metrics calculation for transaction nodes."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
try:
    from .loader import validate_edges, validate_nodes
except ImportError:  # Direct `python src/pipeline.py` entry point.
    from loader import validate_edges, validate_nodes


METRIC_COLUMNS = [
    "gid", "depth", "is_seed", "in_degree", "out_degree", "in_amount", "out_amount",
    "in_tx_count", "out_tx_count", "unique_senders", "unique_receivers", "pass_through",
    "pagerank", "betweenness", "truncated_by_depth",
]


def compute_node_metrics(G: nx.DiGraph, nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """One row per gid; `truncated_by_depth` flags censored leaves explicitly."""
    node = validate_nodes(nodes)[["gid", "depth", "is_seed"]].copy()
    edges = validate_edges(edges, node)
    if not G.is_directed() or G.is_multigraph():
        raise ValueError("Metrics require a simple directed DiGraph")
    if set(G) != set(node.gid) or G.number_of_edges() != len(edges):
        raise ValueError("Graph and source tables have different nodes or edges")
    for edge in edges.itertuples(index=False):
        attrs = G.get_edge_data(edge.src, edge.dst)
        if (attrs is None or attrs.get("n_tx") != edge.n_tx
                or not np.isclose(attrs.get("sum_kzt", np.nan), edge.sum_kzt, rtol=0, atol=0.005)):
            raise ValueError("Graph edge weights disagree with source aggregates")

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
    f = node.merge(deg, on="gid", how="left", validate="one_to_one").join(f, on="gid", validate="one_to_one")
    for c in ("in_degree", "out_degree", "in_tx_count", "out_tx_count", "unique_senders", "unique_receivers"):
        f[c] = f[c].fillna(0).astype("int64")
    for c in ("in_amount", "out_amount"):
        f[c] = f[c].fillna(0.0).astype("float64")

    # At depth four, absent outflow is censored by extraction, not evidence of a true sink.
    f["truncated_by_depth"] = (f.depth >= 4) & (f.out_degree == 0)
    f["pass_through"] = np.divide(
        f.out_amount.to_numpy(),
        f.in_amount.to_numpy(),
        out=np.full(len(f), np.nan),
        where=f.in_amount.to_numpy() > 0
    )
    # Suppress misleading ratios on seeds: their incoming history lies outside the extract.
    f.loc[f.is_seed, "pass_through"] = np.nan

    pr = nx.pagerank(G, weight="sum_kzt") if G.number_of_nodes() else {}
    f["pagerank"] = f.gid.map(pr).fillna(0.0)

    # Weighted betweenness uses inverse transfer volume as path cost.
    for u, v, d in G.edges(data=True):
        d["distance"] = 1.0 / d["sum_kzt"]
    btw = (
        nx.betweenness_centrality(G, weight="distance", normalized=True)
        if len(G) < 10000
        else nx.betweenness_centrality(G, k=min(500, len(G)), weight="distance", seed=42)
    )
    f["betweenness"] = f.gid.map(btw).fillna(0.0)
    return f[METRIC_COLUMNS]


def save_node_metrics(df: pd.DataFrame, output_path: Path | str) -> None:
    missing = set(METRIC_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Metric export missing columns: {sorted(missing)}")
    if df.gid.dtype != np.dtype("int64") or df.gid.isna().any() or not df.gid.is_unique:
        raise ValueError("Metric export requires unique non-null int64 GIDs")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
