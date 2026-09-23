"""Read-only data contract shared by the UI and its checks."""
from pathlib import Path
import os

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(os.environ.get("TRACEFLOW_RESULTS_DIR", str(ROOT / "output")))
DATA_DIR = Path(os.environ.get("TRACEFLOW_DATA_DIR", str(ROOT / "data (1)" / "data")))


@st.cache_data(ttl=20, show_spinner=False)
def read_source(name: str) -> tuple[pd.DataFrame, str | None]:
    path = DATA_DIR / f"{name}.parquet"
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
