"""Graph construction from transaction edges and nodes."""
from __future__ import annotations

import networkx as nx
import pandas as pd
try:
    from .loader import validate_edges, validate_nodes
except ImportError:  # Direct `python src/pipeline.py` entry point.
    from loader import validate_edges, validate_nodes


def build_graph(edges: pd.DataFrame, nodes: pd.DataFrame | None = None) -> nx.DiGraph:
    """Directed, weighted graph. Keep isolated nodes from nodes.parquet if provided."""
    nodes = validate_nodes(nodes) if nodes is not None else None
    edges = validate_edges(edges, nodes)
    G = nx.DiGraph()
    if nodes is not None:
        # itertuples preserves integer IDs, unlike mixed numeric iterrows rows.
        G.add_nodes_from((r.gid, {"depth": int(r.depth), "is_seed": bool(r.is_seed)})
                         for r in nodes.itertuples(index=False))
    for r in edges.itertuples(index=False):
        G.add_edge(r.src, r.dst, sum_kzt=float(r.sum_kzt), n_tx=int(r.n_tx), depth=int(r.depth))
    return G
