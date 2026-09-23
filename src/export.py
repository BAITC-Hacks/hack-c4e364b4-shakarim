"""CSV exports for analyst review."""
from __future__ import annotations

import csv
from pathlib import Path


def write_csv(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export(nodes, clusters, output_dir):
    output_dir = Path(output_dir)
    write_csv(output_dir / "nodes_roles.csv",
              ["gid", "role", "role_score", "cluster_id", "priority_score", "evidence"], nodes)
    write_csv(output_dir / "clusters.csv",
              ["cluster_id", "n_nodes", "n_seed", "sum_kzt_internal", "top_gids", "hypothesis"], clusters)
    ranked = sorted(nodes, key=lambda row: (-float(row["priority_score"]), str(row["gid"])))[:max(20, min(100, len(nodes)))]
    write_csv(output_dir / "top_nodes.csv", ["rank", "gid", "role", "priority_score", "why"],
              [dict(rank=i, gid=row["gid"], role=row["role"], priority_score=row["priority_score"],
                    why=f"Приоритет {float(row['priority_score']):.3f}/1. {row['evidence']}")
               for i, row in enumerate(ranked, 1)])
