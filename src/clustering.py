"""Deterministic weighted Louvain-style local modularity optimization."""
from __future__ import annotations

from collections import defaultdict


def louvain(nodes, edges):
    """Return stable integer communities; edges are (source, target, weight)."""
    graph = {str(n): defaultdict(float) for n in nodes}
    for source, target, weight in edges:
        source, target, weight = str(source), str(target), float(weight)
        if source == target or weight <= 0:
            continue
        graph.setdefault(source, defaultdict(float))[target] += weight
        graph.setdefault(target, defaultdict(float))[source] += weight
    degree = {node: sum(neighbors.values()) for node, neighbors in graph.items()}
    m2 = sum(degree.values())
    if m2 == 0:
        return {node: i + 1 for i, node in enumerate(sorted(graph))}
    community = {node: i for i, node in enumerate(sorted(graph))}
    # Local move step of Louvain. Modularity gain is proportional to k_i,in - k_i*tot_c/m.
    for _ in range(100):
        changed = False
        for node in sorted(graph):
            old = community[node]
            weights_by_comm = defaultdict(float)
            for neighbor, weight in graph[node].items():
                weights_by_comm[community[neighbor]] += weight
            totals = defaultdict(float)
            for member, comm in community.items():
                totals[comm] += degree[member]
            community[node] = -1
            best, best_gain = old, 0.0
            for candidate in sorted(weights_by_comm):
                gain = weights_by_comm[candidate] - degree[node] * totals[candidate] / m2
                if gain > best_gain + 1e-12:
                    best, best_gain = candidate, gain
            community[node] = best
            changed |= best != old
        if not changed:
            break
    groups = defaultdict(list)
    for node, comm in community.items():
        groups[comm].append(node)
    ordered = sorted(groups.values(), key=lambda group: min(group))
    return {node: i + 1 for i, group in enumerate(ordered) for node in group}
