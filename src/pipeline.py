"""Run the complete explainable node analytics pipeline.

Usage from repository root:
    python src/pipeline.py [--data data] [--output-dir output]
    python src/pipeline.py [--input PATH] [--output-dir output]
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
import sys

# Ensure src/ directory is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from clustering import louvain
from export import export
from graph_builder import build_graph
from loader import find_data_dir, load, sanity_check
from metrics import compute_node_metrics, save_node_metrics
from priority import priority_score
from roles import classify, is_seed, key_name, number


GID_NAMES = {"gid", "node_id", "node", "account_id"}
SOURCE_NAMES = {"source", "source_gid", "src", "from_gid", "sender_gid", "from"}
TARGET_NAMES = {"target", "target_gid", "dst", "to_gid", "receiver_gid", "to"}


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


def analyze(rows, edge_rows=None):
    metrics, extracted_edges = {}, []
    for row in rows:
        gid = str(find_value(row, GID_NAMES)).strip()
        source = str(find_value(row, SOURCE_NAMES)).strip()
        target = str(find_value(row, TARGET_NAMES)).strip()
        if source and target:
            extracted_edges.append((source, target, max(0.0, number(row, "amount"))))
        if gid:
            metrics[gid] = row
        elif not (source and target):
            raise ValueError("Для каждой строки CSV требуется gid/node_id или пара source/target.")
    if edge_rows is None:
        edge_rows = extracted_edges
    if not metrics and not edge_rows:
        raise ValueError("Не найдены узлы или транзакционные рёбра.")

    # Derive consistent count and amount metrics from a transaction edge list when present.
    derived = defaultdict(lambda: {"in_amount": 0.0, "out_amount": 0.0, "in_count": 0,
                                   "out_count": 0, "senders": set(), "receivers": set()})
    for source, target, amount in edge_rows:
        derived[source]["out_amount"] += amount
        derived[source]["out_count"] += 1
        derived[source]["receivers"].add(target)
        derived[target]["in_amount"] += amount
        derived[target]["in_count"] += 1
        derived[target]["senders"].add(source)
    all_gids = sorted(set(metrics) | {x for edge in edge_rows for x in edge[:2]})

    # A node-level aggregate file may already contain unique degree counts; preserve those.
    for gid in all_gids:
        if gid not in metrics:
            d = derived[gid]
            metrics[gid] = {"gid": gid, "in_amount": d["in_amount"], "out_amount": d["out_amount"],
                            "in_count": d["in_count"], "out_count": d["out_count"],
                            "unique_senders": len(d["senders"]), "unique_receivers": len(d["receivers"])}
    cluster_map = louvain(all_gids, edge_rows)
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
    for source, target, amount in edge_rows:
        if cluster_map[source] == cluster_map[target]:
            internal[cluster_map[source]] += amount
    clusters = []
    for cluster_id, members in sorted(grouped.items()):
        roles = Counter(row["role"] for row in members)
        dominant, count = roles.most_common(1)[0]
        seed_count = sum(row["_seed"] for row in members)
        if edge_rows:
            hypothesis = (f"Преобладает роль {dominant} ({count}/{len(members)} узлов); "
                          f"обнаружено seed-узлов: {seed_count}.")
        else:
            hypothesis = (f"Связи отсутствуют во входном CSV; сетевой Louvain недоступен. "
                          f"Преобладает роль {dominant} ({count}/{len(members)} узлов); seed: {seed_count}.")
        top = sorted(members, key=lambda row: (-row["priority_score"], str(row["gid"])))[:5]
        clusters.append({"cluster_id": cluster_id, "n_nodes": len(members), "n_seed": seed_count,
                         "sum_kzt_internal": round(internal[cluster_id], 2),
                         "top_gids": ";".join(str(row["gid"]) for row in top), "hypothesis": hypothesis})
    clean_nodes = [{k: v for k, v in row.items() if not k.startswith("_")} for row in node_results]
    for row in clean_nodes:
        row["priority_score"] = round(float(row["priority_score"]), 2)
    return clean_nodes, clusters


def run_from_parquet(data_dir: Path | str, output_dir: Path | str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edges, nodes, tx = load(data_dir)
    sanity_check(edges, nodes, tx)

    G = build_graph(edges, nodes)
    metrics_df = compute_node_metrics(G, nodes, edges)
    save_node_metrics(metrics_df, output_dir / "node_metrics.csv")

    edge_rows = [(str(r.src), str(r.dst), float(r.sum_kzt)) for r in edges.itertuples(index=False)]
    rows = metrics_df.to_dict(orient="records")

    nodes_results, clusters_results = analyze(rows, edge_rows=edge_rows)
    if len(nodes_results) < 20:
        raise ValueError(f"Для файла top_nodes.csv требуется минимум 20 узлов; найдено {len(nodes_results)}.")
    export(nodes_results, clusters_results, output_dir)
    return nodes_results, clusters_results


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, help="Папка с parquet-файлами (edges.parquet, nodes.parquet, transactions.parquet)")
    parser.add_argument("--input", type=Path, help="CSV узлов или транзакций")
    parser.add_argument("--output-dir", type=Path, default=root / "output")
    args = parser.parse_args(argv)

    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir

    try:
        if args.input:
            input_path = args.input if args.input.is_absolute() else root / args.input
            nodes, clusters = analyze(load_input(input_path))
            if len(nodes) < 20:
                raise ValueError(f"Для файла top_nodes.csv требуется минимум 20 узлов; найдено {len(nodes)}.")
            export(nodes, clusters, output_dir)
        elif args.data:
            data_path = args.data if args.data.is_absolute() else root / args.data
            nodes, clusters = run_from_parquet(data_path, output_dir)
        else:
            # Auto-detect parquet directory or existing node_metrics.csv
            parquet_dir = find_data_dir()
            if parquet_dir.exists() and (parquet_dir / "edges.parquet").exists():
                nodes, clusters = run_from_parquet(parquet_dir, output_dir)
            else:
                candidates = [output_dir / "node_metrics.csv", root / "mock/node_metrics.csv"]
                input_path = next((p for p in candidates if p.is_file()), candidates[0])
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
