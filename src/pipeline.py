"""Analyze precomputed per-node metrics without rebuilding the transaction graph.

Usage from repository root: python3 src/pipeline.py [--input PATH] [--output-dir output]
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import sys

from export import export
from priority import priority_score
from roles import classify, is_seed, key_name


GID_NAMES = {"gid", "node_id", "node", "account_id"}


def find_value(row, names):
    return next((v for k, v in row.items() if key_name(k) in names), "")


def load_input(path):
    if not path.is_file():
        raise FileNotFoundError(f"Не найден CSV входных метрик: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV пуст или не содержит заголовков: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV не содержит строк данных: {path}")
    return rows


def analyze(rows):
    metrics = {}
    for row in rows:
        gid = str(find_value(row, GID_NAMES)).strip()
        if not gid:
            raise ValueError("Ожидается node_metrics.csv: каждая строка должна содержать gid.")
        if gid in metrics:
            raise ValueError(f"В node_metrics.csv повторяется gid={gid}; ожидается одна строка на узел.")
        metrics[gid] = row
    if not metrics:
        raise ValueError("node_metrics.csv не содержит строк с узлами.")

    all_gids = sorted(metrics)
    # Aggregated node metrics do not contain transaction endpoints. Do not construct an
    # inferred graph: until community IDs are supplied, use explicit singleton fallbacks.
    cluster_map = {gid: index for index, gid in enumerate(all_gids, start=1)}
    node_results = []
    for gid in all_gids:
        row = metrics[gid]
        role, role_score, evidence = classify(row)
        score = priority_score(row)
        node_results.append({"gid": gid, "role": role, "role_score": role_score,
                             "cluster_id": cluster_map[gid], "priority_score": score, "evidence": evidence,
                             "_seed": is_seed(row)})
    grouped = defaultdict(list)
    for row in node_results:
        grouped[row["cluster_id"]].append(row)
    internal = defaultdict(float)
    clusters = []
    for cluster_id, members in sorted(grouped.items()):
        roles = Counter(row["role"] for row in members)
        dominant, count = roles.most_common(1)[0]
        seed_count = sum(row["_seed"] for row in members)
        hypothesis = (f"В node_metrics.csv нет исходных связей; узел оставлен в отдельном кластере. "
                      f"Роль: {dominant}; seed-узлов: {seed_count}.")
        top = sorted(members, key=lambda row: (-row["priority_score"], row["gid"]))[:5]
        clusters.append({"cluster_id": cluster_id, "n_nodes": len(members), "n_seed": seed_count,
                         "sum_kzt_internal": round(internal[cluster_id], 2),
                         "top_gids": ";".join(row["gid"] for row in top), "hypothesis": hypothesis})
    clean_nodes = [{k: v for k, v in row.items() if not k.startswith("_")} for row in node_results]
    for row in clean_nodes:
        row["priority_score"] = round(float(row["priority_score"]), 2)
    return clean_nodes, clusters


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="node_metrics.csv (одна агрегированная строка на gid)")
    parser.add_argument("--output-dir", type=Path, default=root / "output")
    args = parser.parse_args(argv)
    if args.input:
        input_path = args.input if args.input.is_absolute() else root / args.input
    else:
        candidates = [root / "output/node_metrics.csv", root / "mock/node_metrics.csv"]
        input_path = next((p for p in candidates if p.is_file()), candidates[0])
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    try:
        nodes, clusters = analyze(load_input(input_path))
        if len(nodes) < 20:
            raise ValueError(f"Для файла top_nodes.csv требуется минимум 20 узлов; найдено {len(nodes)}.")
        export(nodes, clusters, output_dir)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Ошибка аналитического pipeline: {exc}", file=sys.stderr)
        return 2
    print(f"Готово: {len(nodes)} узлов, {len(clusters)} кластеров; CSV сохранены в {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
