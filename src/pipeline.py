"""Run Parquet analytics or analyze precomputed node metrics and optional edges.

Usage from the repository root:
    python src/pipeline.py --data data --output-dir output
    python src/pipeline.py --input output/node_metrics.csv --edges output/edges.csv
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

from clustering import louvain
from export import export, write_csv
from priority import priority_reason, priority_score
from roles import ALIASES, classify, is_seed, key_name


GID_NAMES = {"gid", "node_id", "node", "account_id"}
SOURCE_NAMES = {"source", "source_gid", "src", "from_gid", "sender_gid", "from"}
TARGET_NAMES = {"target", "target_gid", "dst", "to_gid", "receiver_gid", "to"}


def find_value(row, names):
    return next((v for k, v in row.items() if key_name(k) in names), "")


def load_input(path, *, allow_empty=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Не найден CSV входных метрик: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV пуст или не содержит заголовков: {path}")
        names = [key_name(name) for name in reader.fieldnames]
        if not all(names) or len(set(names)) != len(names):
            raise ValueError(f"CSV содержит пустые или повторные заголовки: {path}")
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Число полей в строке CSV не совпадает с заголовком: {path}")
    if not rows and not allow_empty:
        raise ValueError(f"CSV не содержит строк данных: {path}")
    return rows


def metric_number(value, label):
    try:
        result = float(str(value).strip().replace(" ", "").replace(",", "."))
    except (ValueError, TypeError):
        raise ValueError(f"{label}: требуется конечное неотрицательное число.") from None
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{label}: требуется конечное неотрицательное число.")
    return result


def validate_metrics(row, gid):
    """Reject corrupt CSV values rather than interpreting them as zero activity."""
    normalized = {key_name(key): value for key, value in row.items()}
    for name in ("in_amount", "out_amount", "unique_senders", "unique_receivers"):
        present = next((alias for alias in ALIASES[name] if alias in normalized), None)
        if present is None:
            raise ValueError(f"gid={gid}: отсутствует обязательная метрика {name}.")
        value = metric_number(normalized[present], f"gid={gid}, {name}")
        if name.startswith("unique_") and not value.is_integer():
            raise ValueError(f"gid={gid}, {name}: ожидается целый счётчик.")
    for name in ("depth", "in_degree", "out_degree", "in_count", "out_count"):
        present = next((alias for alias in ALIASES.get(name, (name,)) if alias in normalized), None)
        if present is not None:
            value = metric_number(normalized[present], f"gid={gid}, {name}")
            if not value.is_integer():
                raise ValueError(f"gid={gid}, {name}: ожидается целый счётчик.")
    for name in ALIASES["seed"]:
        if name in normalized and str(normalized[name]).strip().lower() not in {
            "true", "false", "0", "1", "0.0", "1.0", "yes", "no", "да", "нет", "seed"
        }:
            raise ValueError(f"gid={gid}: некорректный флаг {name}.")


def parse_edges(rows, known_gids):
    """Parse directed transaction edges for an undirected weighted Louvain projection."""
    edges = []
    for index, row in enumerate(rows, start=2):
        source = str(find_value(row, SOURCE_NAMES)).strip()
        target = str(find_value(row, TARGET_NAMES)).strip()
        if not source or not target:
            raise ValueError(f"В edges CSV строка {index} должна содержать source и target.")
        if source not in known_gids or target not in known_gids:
            raise ValueError(f"В edges CSV строка {index} ссылается на gid вне node_metrics.csv.")
        weight = metric_number(find_value(row, ALIASES["amount"]), f"edges CSV, строка {index}, вес")
        edges.append((source, target, weight))
    return edges


def analyze(rows, edge_rows=None):
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
    for gid, row in metrics.items():
        validate_metrics(row, gid)

    all_gids = sorted(metrics)
    edge_rows = [(str(source), str(target), metric_number(weight, "Вес ребра"))
                 for source, target, weight in (edge_rows or [])]
    # louvain() aggregates directed weights into an undirected weighted projection.
    cluster_map = louvain(all_gids, edge_rows) if edge_rows else {
        gid: index for index, gid in enumerate(all_gids, start=1)
    }
    node_results = []
    for gid in all_gids:
        row = metrics[gid]
        role, role_score, evidence = classify(row)
        score = priority_score(row)
        node_results.append({"gid": gid, "role": role, "role_score": role_score,
                             "cluster_id": cluster_map[gid], "priority_score": score, "evidence": evidence,
                             "why": priority_reason(role, score, evidence, row),
                             "_seed": is_seed(row)})
    grouped = defaultdict(list)
    for row in node_results:
        grouped[row["cluster_id"]].append(row)
    internal = defaultdict(float)
    for source, target, weight in edge_rows:
        if cluster_map[source] == cluster_map[target]:
            internal[cluster_map[source]] += weight
    clusters = []
    for cluster_id, members in sorted(grouped.items()):
        roles = Counter(row["role"] for row in members)
        dominant, count = roles.most_common(1)[0]
        seed_count = sum(row["_seed"] for row in members)
        if edge_rows:
            purpose = {"terminal": "возможное удержание средств", "transit": "возможный транзит",
                       "consolidator": "возможная консолидация", "distributor": "возможное распределение",
                       "coordinator": "возможная координация", "peripheral": "назначение не определено"}[dominant]
            hypothesis = (f"Гипотеза: {purpose}; {dominant} у {count}/{len(members)} узлов, "
                          f"seed: {seed_count}. Основа — Louvain по объёму переводов.")
        else:
            hypothesis = (f"В node_metrics.csv нет исходных связей; узел оставлен в отдельном кластере. "
                          f"Роль: {dominant}; seed-узлов: {seed_count}.")
        top = sorted(members, key=lambda row: (-row["priority_score"], row["gid"]))[:5]
        clusters.append({"cluster_id": cluster_id, "n_nodes": len(members), "n_seed": seed_count,
                         "sum_kzt_internal": round(internal[cluster_id], 2),
                         "top_gids": ";".join(row["gid"] for row in top), "hypothesis": hypothesis})
    clean_nodes = [{k: v for k, v in row.items() if not k.startswith("_")} for row in node_results]
    for row in clean_nodes:
        row["priority_score"] = round(float(row["priority_score"]), 3)
    return clean_nodes, clusters


def publish_staged(stage, output_dir):
    """Publish only after every calculation and report validation has succeeded."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # The context binds this generation of reports to its source dataset.
    # Publish it last, so an interrupted publication cannot validate mixed files.
    for path in sorted(Path(stage).iterdir(), key=lambda path: (path.name == "analysis_context.json", path.name)):
        os.replace(path, output_dir / path.name)


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_context(stage, data_dir=None):
    """Bind the exported reports to raw data, or mark them as metrics-only."""
    stage = Path(stage)
    source = Path(data_dir).resolve() if data_dir is not None else None
    context = {
        "schema_version": 1,
        "mode": "parquet" if source else "metrics",
        "data_dir": str(source) if source else None,
        "source_sha256": {name: file_sha256(source / name) for name in (
            "nodes.parquet", "edges.parquet", "transactions.parquet"
        )} if source else {},
        "output_sha256": {name: file_sha256(stage / name) for name in (
            "node_metrics.csv", "edges.csv", "nodes_roles.csv", "clusters.csv", "top_nodes.csv"
        )},
    }
    (stage / "analysis_context.json").write_text(
        json.dumps(context, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def canonical_metrics(rows):
    """Keep supplied measurements, with canonical names for downstream UI."""
    canonical = []
    for row in rows:
        normalized = {key_name(key): value for key, value in row.items()}
        for name, aliases in ALIASES.items():
            if name == "amount":
                continue
            present = next((alias for alias in aliases if alias in normalized), None)
            if present is not None:
                target = {"seed": "is_seed", "in_count": "in_tx_count", "out_count": "out_tx_count"}.get(name, name)
                value = normalized[present]
                for alias in aliases:
                    if alias not in {"in_degree", "out_degree"}:
                        normalized.pop(alias, None)
                normalized[target] = is_seed(row) if name == "seed" else value
        gid = str(find_value(row, GID_NAMES)).strip()
        for alias in GID_NAMES:
            normalized.pop(alias, None)
        canonical.append({"gid": gid, **normalized})
    return canonical


def write_input_exports(stage, metric_rows, source_edges, parsed_edges):
    rows = canonical_metrics(metric_rows)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    write_csv(stage / "node_metrics.csv", columns, rows)
    edges = []
    for original, (source, target, weight) in zip(source_edges, parsed_edges):
        extras = {key_name(key): value for key, value in original.items()
                  if key_name(key) not in SOURCE_NAMES | TARGET_NAMES | set(ALIASES["amount"])}
        edges.append({"src": source, "dst": target, "sum_kzt": weight, **extras})
    columns = list(dict.fromkeys(["src", "dst", "sum_kzt"] + [key for edge in edges for key in edge]))
    write_csv(stage / "edges.csv", columns, edges)


def run_from_parquet(data_dir: Path | str, output_dir: Path | str):
    """Connect validated graph metrics with the analytics and UI CSV contracts."""
    from graph_builder import build_graph
    from loader import load, sanity_check
    from metrics import compute_node_metrics, save_node_metrics

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    edges, nodes, transactions = load(data_dir)
    sanity_check(edges, nodes, transactions)
    graph = build_graph(edges, nodes)
    metrics_df = compute_node_metrics(graph, nodes, edges)
    edge_rows = [(str(row.src), str(row.dst), float(row.sum_kzt))
                 for row in edges.itertuples(index=False)]
    node_results, clusters = analyze(metrics_df.to_dict(orient="records"), edge_rows=edge_rows)
    if len(node_results) < 20:
        raise ValueError(f"Для файла top_nodes.csv требуется минимум 20 узлов; найдено {len(node_results)}.")
    with tempfile.TemporaryDirectory(prefix=".traceflow-", dir=output_dir.parent) as directory:
        stage = Path(directory)
        save_node_metrics(metrics_df, stage / "node_metrics.csv")
        edges.to_csv(stage / "edges.csv", index=False)
        export(node_results, clusters, stage)
        write_context(stage, data_dir)
        publish_staged(stage, output_dir)
    return node_results, clusters


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--data", type=Path, help="Папка с nodes.parquet, edges.parquet и transactions.parquet")
    inputs.add_argument("--input", type=Path, help="node_metrics.csv (одна агрегированная строка на gid)")
    parser.add_argument("--edges", type=Path, help="опциональный CSV source,target,amount для Louvain")
    parser.add_argument("--output-dir", type=Path, default=root / "output")
    args = parser.parse_args(argv)
    if args.data and args.edges:
        parser.error("--edges используется с --input; режим --data читает edges.parquet.")
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    try:
        data_path = None
        if args.data:
            data_path = args.data if args.data.is_absolute() else root / args.data
        elif not args.input and not args.edges:
            from loader import find_data_dir
            candidate = find_data_dir()
            if (candidate / "edges.parquet").is_file():
                data_path = candidate
        if data_path is not None:
            nodes, clusters = run_from_parquet(data_path, output_dir)
        else:
            input_path = args.input or (output_dir / "node_metrics.csv")
            if not input_path.is_absolute():
                input_path = root / input_path
            metric_rows = load_input(input_path)
            gids = {str(find_value(row, GID_NAMES)).strip() for row in metric_rows}
            edge_rows = []
            source_edges = []
            if args.edges:
                edges_path = args.edges if args.edges.is_absolute() else root / args.edges
                source_edges = load_input(edges_path, allow_empty=True)
                edge_rows = parse_edges(source_edges, gids)
            nodes, clusters = analyze(metric_rows, edge_rows=edge_rows)
            if len(nodes) < 20:
                raise ValueError(f"Для файла top_nodes.csv требуется минимум 20 узлов; найдено {len(nodes)}.")
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".traceflow-", dir=output_dir.parent) as directory:
                stage = Path(directory)
                export(nodes, clusters, stage)
                write_input_exports(stage, metric_rows, source_edges, edge_rows)
                write_context(stage)
                publish_staged(stage, output_dir)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"Ошибка аналитического pipeline: {exc}", file=sys.stderr)
        return 2
    print(f"Готово: {len(nodes)} узлов, {len(clusters)} кластеров; CSV сохранены в {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
