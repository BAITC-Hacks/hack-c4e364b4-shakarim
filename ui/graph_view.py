"""A small, read-only neighbourhood view for a selected client."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import streamlit as st

from components import role_color


def load_edges(candidates: Iterable[Path]) -> tuple[pd.DataFrame, Path | None]:
    """Load an edge file solely for rendering existing links, never for scoring."""
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".parquet":
                return pd.read_parquet(path), path
            return pd.read_csv(path), path
        except Exception as error:  # pragma: no cover - shown to the analyst
            st.warning(f"Не удалось прочитать связи из {path.name}: {error}")
    return pd.DataFrame(columns=["src", "dst"]), None


def _dot_value(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def _role_by_gid(nodes: pd.DataFrame) -> dict[str, object]:
    if not {"gid", "role"}.issubset(nodes.columns):
        return {}
    return dict(zip(nodes["gid"].astype(str), nodes["role"]))


def render_client_connections(selected_gid: object, edges: pd.DataFrame, nodes: pd.DataFrame) -> None:
    """Render observed incoming and outgoing links around one gid."""
    st.subheader("Наблюдаемые связи клиента")
    if not {"src", "dst"}.issubset(edges.columns):
        st.info("Файл со связями не найден. Роли и оценки продолжают читаться из готовых CSV.")
        return

    selected = str(selected_gid)
    view = edges.copy()
    view["_src"] = view["src"].astype(str)
    view["_dst"] = view["dst"].astype(str)
    incident = view.loc[(view["_src"] == selected) | (view["_dst"] == selected)].copy()
    if incident.empty:
        st.info("В доступном файле связей для этого клиента нет наблюдаемых переводов.")
        return

    # The cap keeps the screen useful for fan-out nodes; it is not an analytics score.
    order_column = "sum_kzt" if "sum_kzt" in incident.columns else None
    if order_column:
        incident = incident.sort_values(order_column, ascending=False)
    incident = incident.head(40)

    roles = _role_by_gid(nodes)
    gids = set(incident["_src"]) | set(incident["_dst"])
    lines = ["digraph neighbourhood {", "rankdir=LR;", "graph [bgcolor=\"transparent\", pad=0.2];", "node [shape=ellipse, style=filled, fontname=Arial, fontcolor=white];"]
    for gid in gids:
        is_selected = gid == selected
        size = "1.65" if is_selected else "1.05"
        lines.append(
            f'"{_dot_value(gid)}" [label="{_dot_value(gid)}", fillcolor="{role_color(roles.get(gid))}", width={size}];'
        )
    for edge in incident.itertuples(index=False):
        src, dst = str(edge.src), str(edge.dst)
        label = ""
        if hasattr(edge, "sum_kzt") and pd.notna(edge.sum_kzt):
            label = f' [label="{float(edge.sum_kzt):,.0f} KZT", fontsize=9]'
        lines.append(f'"{_dot_value(src)}" -> "{_dot_value(dst)}"{label};')
    lines.append("}")
    st.graphviz_chart("\n".join(lines), use_container_width=True)
    st.caption(f"Показано до 40 наблюдаемых связей клиента {selected}. Цвета обозначают роли из nodes_roles.csv.")
