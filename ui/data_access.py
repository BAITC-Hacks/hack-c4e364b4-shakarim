"""Read-only data contract shared by the UI and its checks."""
from pathlib import Path
import hashlib
import json
import os

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("TRACEFLOW_RESULTS_DIR", str(ROOT / "output")))
DATA_DIR = Path(os.environ.get("TRACEFLOW_DATA_DIR", str(ROOT / "data")))
SOURCE_FILES = ("nodes.parquet", "edges.parquet", "transactions.parquet")
OUTPUT_FILES = ("node_metrics.csv", "edges.csv", "nodes_roles.csv", "clusters.csv", "top_nodes.csv")


def file_signature(path: Path) -> tuple[int, int] | None:
    """Include file changes in caches even before their short TTL expires."""
    try:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def validate_analysis_context(directory: Path, data_directory: Path) -> tuple[bool, str | None]:
    """Authorize raw-data enrichment only for the exact dataset behind reports.

    Cache invalidates when the manifest, any source file, or any output changes.
    Matching GIDs alone cannot establish that transactions belong to this run.
    """
    directory, data_directory = Path(directory).resolve(), Path(data_directory).resolve()
    paths = [directory / "analysis_context.json", *(directory / name for name in OUTPUT_FILES),
             *(data_directory / name for name in SOURCE_FILES)]
    signatures = tuple((str(path), file_signature(path)) for path in paths)
    return _validate_analysis_context(str(directory), str(data_directory), signatures)


@st.cache_data(ttl=20, show_spinner=False)
def _validate_analysis_context(directory: str, data_directory: str, signatures: tuple) -> tuple[bool, str | None]:
    """Hash checks are read-only; ``signatures`` is part of the cache key."""
    results, source = Path(directory), Path(data_directory)
    unavailable = "История переводов из исходных данных не подключена: "
    manifest = results / "analysis_context.json"
    if not manifest.is_file():
        return False, unavailable + "нет подтверждения, что Parquet и аналитические CSV относятся к одному запуску."
    try:
        context = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(context, dict) or context.get("schema_version") != 1:
            raise ValueError("неподдерживаемый формат analysis_context.json")
        if context.get("mode") == "metrics":
            return False, unavailable + "эти результаты рассчитаны из CSV метрик; исходные транзакции к ним не привязаны."
        if context.get("mode") != "parquet" or not isinstance(context.get("data_dir"), str):
            raise ValueError("неверный режим или путь в analysis_context.json")
        if Path(context["data_dir"]).resolve() != source:
            raise ValueError("папка Parquet отличается от использованной при расчёте")
        for root, names, hashes in ((source, SOURCE_FILES, context.get("source_sha256")),
                                    (results, OUTPUT_FILES, context.get("output_sha256"))):
            if not isinstance(hashes, dict) or set(names) != set(hashes):
                raise ValueError("неполный список контрольных сумм в analysis_context.json")
            for name in names:
                digest = hashlib.sha256()
                with (root / name).open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != hashes[name]:
                    raise ValueError(f"{name} изменён после расчёта")
    except (OSError, ValueError, TypeError) as error:
        return False, unavailable + str(error) + ". Пересчитайте аналитику для выбранной папки данных."
    return True, None


def read_source(name: str) -> tuple[pd.DataFrame, str | None]:
    path = DATA_DIR / f"{name}.parquet"
    return _read_source(name, str(path), file_signature(path))


@st.cache_data(ttl=20, show_spinner=False)
def _read_source(name: str, filename: str, signature: tuple | None) -> tuple[pd.DataFrame, str | None]:
    path = Path(filename)
    if not path.exists():
        return pd.DataFrame(), f"Не найден {path.name}"
    try:
        frame = pd.read_parquet(path)
        required = {"nodes": {"gid", "depth", "is_seed"},
                    "transactions": {"src", "dst", "date", "sum_kzt"}}[name]
        if not required.issubset(frame):
            return pd.DataFrame(), f"{path.name}: неверный набор колонок"
        if name == "transactions":
            frame["date"] = pd.to_datetime(frame["date"], errors="raise")
            frame["sum_kzt"] = pd.to_numeric(frame["sum_kzt"], errors="raise")
            if frame[list(required)].isna().any().any() or (frame.sum_kzt < 0).any():
                raise ValueError("Invalid transaction values")
        return frame, None
    except (ValueError, OSError, ImportError, KeyError) as error:
        return pd.DataFrame(), f"Не удалось прочитать {path.name}: {type(error).__name__}"


def raw_profiles(source: pd.DataFrame) -> pd.DataFrame:
    """Expose observed clients without inventing model output."""
    result = source.copy()
    result["role"] = "unassigned"
    result["role_score"] = float("nan")
    result["priority_score"] = float("nan")
    result["cluster_id"] = "—"
    result["evidence"] = "Роль и приоритет пока не рассчитаны. Ниже доступны исходные переводы клиента."
    return result
