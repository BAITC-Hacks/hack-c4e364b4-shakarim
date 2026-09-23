"""Full weighted Louvain on an undirected projection; metrics stay directed.

Parallel and reciprocal transfers are summed into one undirected edge. Self
transfers contribute to financial totals, but not to community discovery.
Sorted insertion and a fixed seed make the result reproducible for a dataset.
"""
from __future__ import annotations

from collections import defaultdict
import math

import networkx as nx


def weighted_projection(nodes, edges):
    """Build only the clustering projection, without recalculating node metrics."""
    graph = nx.Graph()
    graph.add_nodes_from(sorted({str(node) for node in nodes}))
    amounts = defaultdict(list)
    for source, target, weight in edges:
        source, target = str(source), str(target)
        if source not in graph or target not in graph:
            raise ValueError("Ребро ссылается на gid вне node_metrics.csv.")
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Вес ребра должен быть конечным неотрицательным числом.")
        if source != target and weight > 0:
            amounts[tuple(sorted((source, target)))].append(weight)
    for (source, target), weights in sorted(amounts.items()):
        graph.add_edge(source, target, weight=math.fsum(sorted(weights)))
    return graph


def louvain(nodes, edges):
    """Return stable IDs using multilevel Louvain (resolution=1, seed=42)."""
    graph = weighted_projection(nodes, edges)
    isolates = [{node} for node in nx.isolates(graph)]
    active = graph.subgraph([node for node, degree in graph.degree() if degree]).copy()
    communities = (
        nx.community.louvain_communities(active, weight="weight", resolution=1, seed=42)
        if active.number_of_edges() else []
    )
    ordered = sorted(communities + isolates, key=lambda group: min(group))
    return {node: index for index, group in enumerate(ordered, start=1) for node in sorted(group)}
