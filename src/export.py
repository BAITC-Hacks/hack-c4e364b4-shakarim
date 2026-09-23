"""CSV exports for analyst review."""
from __future__ import annotations

import csv
from collections import Counter
import math
import os
from pathlib import Path
import tempfile

from roles import MAX_EVIDENCE_LENGTH, ROLE_NAMES

NODES_ROLES_COLUMNS = ("gid", "role", "role_score", "cluster_id", "priority_score", "evidence")
CLUSTERS_COLUMNS = ("cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis")
TOP_NODES_COLUMNS = ("rank", "gid", "role", "priority_score", "why")


def validate_reports(nodes, clusters):
    if len(nodes) < 20:
        raise ValueError("Для top_nodes.csv требуется минимум 20 уникальных узлов.")
    gids, membership = set(), Counter()
    for row in nodes:
        if any(key not in row or str(row[key]).strip() == "" for key in NODES_ROLES_COLUMNS):
            raise ValueError("В nodes_roles.csv не заполнены обязательные поля.")
        gid = str(row["gid"])
        if gid in gids:
            raise ValueError(f"Повторный gid в результатах: {gid}")
        gids.add(gid)
        membership[row["cluster_id"]] += 1
        if row["role"] not in ROLE_NAMES:
            raise ValueError(f"Неизвестная роль для gid={gid}.")
        for field in ("role_score", "priority_score"):
            value = float(row[field])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{field} для gid={gid} должен быть в диапазоне 0–1.")
        if not isinstance(row["evidence"], str) or not 1 <= len(row["evidence"]) <= MAX_EVIDENCE_LENGTH:
            raise ValueError(f"evidence для gid={gid} должен содержать 1–200 символов.")
    seen = set()
    for row in clusters:
        if any(key not in row or str(row[key]).strip() == "" for key in CLUSTERS_COLUMNS):
            raise ValueError("В clusters.csv не заполнены обязательные поля.")
        cluster_id = row["cluster_id"]
        if cluster_id in seen or cluster_id not in membership:
            raise ValueError("cluster_id должен быть уникальным и существовать в nodes_roles.csv.")
        seen.add(cluster_id)
        if row["n_nodes"] != membership[cluster_id] or not 0 <= row["n_seed"] <= row["n_nodes"]:
            raise ValueError("Размер кластера или число seed не согласованы с узлами.")
        amount = float(row["sum_kzt_internal"])
        if not math.isfinite(amount) or amount < 0:
            raise ValueError("Внутренний оборот кластера должен быть конечным и неотрицательным.")
    if seen != set(membership):
        raise ValueError("Не все кластеры из nodes_roles.csv представлены в clusters.csv.")


def write_csv(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export(nodes, clusters, output_dir):
    nodes, clusters = list(nodes), list(clusters)
    validate_reports(nodes, clusters)
    output_dir = Path(output_dir)
    ranked = sorted(nodes, key=lambda row: (-float(row["priority_score"]), str(row["gid"])))[:100]
    top = [dict(rank=index, gid=row["gid"], role=row["role"], priority_score=row["priority_score"],
                why=row.get("why") or f"Приоритет {float(row['priority_score']):.3f}/1. {row['evidence']}")
           for index, row in enumerate(ranked, 1)]
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".traceflow-reports-", dir=output_dir.parent) as directory:
        stage = Path(directory)
        write_csv(stage / "nodes_roles.csv", NODES_ROLES_COLUMNS, nodes)
        write_csv(stage / "clusters.csv", CLUSTERS_COLUMNS, clusters)
        write_csv(stage / "top_nodes.csv", TOP_NODES_COLUMNS, top)
        output_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted(stage.iterdir()):
            os.replace(path, output_dir / path.name)
