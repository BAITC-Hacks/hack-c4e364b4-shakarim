"""Data loader for graph parquet files."""
from __future__ import annotations

from pathlib import Path
import pandas as pd


def find_data_dir(data_dir: Path | str | None = None) -> Path:
    if data_dir is not None:
        p = Path(data_dir)
        if p.exists() and (p / "edges.parquet").exists():
            return p
    candidates = [
        Path("data"),
        Path("data (1)/data"),
        Path("../data"),
        Path(__file__).resolve().parents[1] / "data",
        Path(__file__).resolve().parents[1] / "data (1)" / "data",
    ]
    for c in candidates:
        if c.exists() and (c / "edges.parquet").exists():
            return c
    return Path(data_dir or "data")


def load(data_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load edges, nodes, and transactions parquet files."""
    resolved_dir = find_data_dir(data_dir)
    edges_path = resolved_dir / "edges.parquet"
    nodes_path = resolved_dir / "nodes.parquet"
    tx_path = resolved_dir / "transactions.parquet"

    if not edges_path.exists():
        raise FileNotFoundError(f"edges.parquet not found in {resolved_dir}")
    if not nodes_path.exists():
        raise FileNotFoundError(f"nodes.parquet not found in {resolved_dir}")
    if not tx_path.exists():
        raise FileNotFoundError(f"transactions.parquet not found in {resolved_dir}")

    edges = pd.read_parquet(edges_path)
    nodes = pd.read_parquet(nodes_path)
    tx = pd.read_parquet(tx_path)
    if "date" in tx.columns:
        tx["date"] = pd.to_datetime(tx["date"])
    return edges, nodes, tx


def sanity_check(edges: pd.DataFrame, nodes: pd.DataFrame, tx: pd.DataFrame) -> set:
    """Data integrity sanity checks before graph and metric processing."""
    in_edges = set(edges.src) | set(edges.dst)
    orphans = set(nodes.gid) - in_edges
    return orphans
