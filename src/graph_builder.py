"""Graph construction from transaction edges and nodes."""
from __future__ import annotations

import networkx as nx
import pandas as pd


def build_graph(edges: pd.DataFrame, nodes: pd.DataFrame | None = None) -> nx.DiGraph:
    """Directed, weighted graph. Keep isolated nodes from nodes.parquet if provided."""
    G = nx.DiGraph()
    if nodes is not None and "gid" in nodes.columns:
        G.add_nodes_from(nodes.gid.tolist())
    for r in edges.itertuples(index=False):
        G.add_edge(r.src, r.dst, sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx), depth=int(r.depth))
    return G
