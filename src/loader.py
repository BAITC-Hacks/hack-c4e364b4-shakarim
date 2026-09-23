"""Load and validate transaction data without losing signed int64 identifiers."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


NODE_COLUMNS = {"gid", "depth", "is_seed"}
EDGE_COLUMNS = {"src", "dst", "sum_kzt", "n_tx", "depth"}
TX_COLUMNS = {"src", "dst", "date", "sum_kzt"}


def _require_columns(frame: pd.DataFrame, name: str, required: set[str]) -> None:
    if not frame.columns.is_unique:
        raise ValueError(f"{name}: duplicate column names")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")
    if frame[list(required)].isna().any().any():
        raise ValueError(f"{name}: null values in required columns")


def _int64(values: pd.Series, name: str, minimum: int | None = None) -> pd.Series:
    # Casting a float/string ID back to int cannot restore digits already lost.
    if not pd.api.types.is_integer_dtype(values.dtype):
        raise ValueError(f"{name}: integer dtype required; float/string identifiers are unsafe")
    limit = np.iinfo(np.int64)
    if not values.empty and (int(values.min()) < limit.min or int(values.max()) > limit.max):
        raise ValueError(f"{name}: value outside int64 range")
    if minimum is not None and (values < minimum).any():
        raise ValueError(f"{name}: values must be >= {minimum}")
    return values.astype("int64")


def _amounts(frame: pd.DataFrame, name: str) -> None:
    values = frame["sum_kzt"]
    if (not pd.api.types.is_numeric_dtype(values.dtype)
            or pd.api.types.is_bool_dtype(values.dtype)):
        raise ValueError(f"{name}.sum_kzt: numeric amounts required")
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(f"{name}.sum_kzt: amounts must be finite and positive")


def validate_nodes(nodes: pd.DataFrame) -> pd.DataFrame:
    """Return a typed copy; reject missing/duplicate nodes and unsafe IDs."""
    _require_columns(nodes, "nodes", NODE_COLUMNS)
    if nodes.empty or not nodes.gid.is_unique:
        raise ValueError("nodes.gid: a nonempty unique identifier list is required")
    result = nodes.copy()
    result["gid"] = _int64(nodes.gid, "nodes.gid")
    result["depth"] = _int64(nodes.depth, "nodes.depth", minimum=0)
    if not (pd.api.types.is_bool_dtype(nodes.is_seed.dtype)
            or pd.api.types.is_integer_dtype(nodes.is_seed.dtype)) or not nodes.is_seed.isin([0, 1]).all():
        raise ValueError("nodes.is_seed: boolean or integer 0/1 values required")
    result["is_seed"] = nodes.is_seed.astype(bool)
    return result


def validate_edges(edges: pd.DataFrame, nodes: pd.DataFrame | None = None) -> pd.DataFrame:
    """Validate one aggregate per directed pair; never silently overwrite edges."""
    _require_columns(edges, "edges", EDGE_COLUMNS)
    result = edges.copy()
    for column in ("src", "dst"):
        result[column] = _int64(edges[column], f"edges.{column}")
    result["n_tx"] = _int64(edges.n_tx, "edges.n_tx", minimum=1)
    result["depth"] = _int64(edges.depth, "edges.depth", minimum=0)
    _amounts(edges, "edges")
    if edges.duplicated(["src", "dst"]).any():
        raise ValueError("edges: duplicate directed src/dst pairs")
    if nodes is not None and (set(result.src) | set(result.dst)) - set(nodes.gid):
        raise ValueError("edges: endpoint missing from nodes.gid")
    return result


def find_data_dir(data_dir: Path | str | None = None) -> Path:
    if data_dir is not None:
        # A typo in an explicitly requested path must fail, not select another dataset.
        return Path(data_dir)
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

    nodes = validate_nodes(pd.read_parquet(nodes_path))
    edges = validate_edges(pd.read_parquet(edges_path), nodes)
    tx = pd.read_parquet(tx_path)
    _require_columns(tx, "transactions", TX_COLUMNS)
    for column in ("src", "dst"):
        tx[column] = _int64(tx[column], f"transactions.{column}")
    try:
        tx["date"] = pd.to_datetime(tx["date"])
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("transactions.date: invalid transaction date") from exc
    sanity_check(edges, nodes, tx)
    return edges, nodes, tx


def sanity_check(edges: pd.DataFrame, nodes: pd.DataFrame, tx: pd.DataFrame) -> set:
    """Verify schemas, all endpoints, and each pair's amount AND transaction count."""
    nodes = validate_nodes(nodes)
    edges = validate_edges(edges, nodes)
    _require_columns(tx, "transactions", TX_COLUMNS)
    for column in ("src", "dst"):
        _int64(tx[column], f"transactions.{column}")
    _amounts(tx, "transactions")
    try:
        dates = pd.to_datetime(tx.date)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("transactions.date: invalid transaction date") from exc
    if dates.isna().any():
        raise ValueError("transactions.date: missing transaction date")
    known = set(nodes.gid)
    if (set(tx.src) | set(tx.dst)) - known:
        raise ValueError("transactions: endpoint missing from nodes.gid")
    aggregate = tx.groupby(["src", "dst"], as_index=False).agg(
        tx_amount=("sum_kzt", "sum"), tx_count=("sum_kzt", "size")
    )
    compared = edges.merge(aggregate, on=["src", "dst"], how="outer", indicator=True, validate="one_to_one")
    if not compared["_merge"].eq("both").all():
        raise ValueError("edges and transactions have different directed pairs")
    # Half a cent allows float summation noise while rejecting a missing cent;
    # rtol=0 avoids accepting large absolute errors on high-volume edges.
    if not np.allclose(compared.sum_kzt, compared.tx_amount, atol=0.005, rtol=0):
        raise ValueError("edges.sum_kzt disagrees with transaction aggregates")
    if not compared.n_tx.eq(compared.tx_count).all():
        raise ValueError("edges.n_tx disagrees with transaction counts")
    in_edges = set(edges.src) | set(edges.dst)
    orphans = set(nodes.gid) - in_edges
    return orphans
