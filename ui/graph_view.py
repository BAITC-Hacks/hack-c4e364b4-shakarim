"""Animated, read-only neighbourhood canvas for the selected customer."""

from __future__ import annotations

from base64 import b64encode
from html import escape
from math import cos, pi, sin
from pathlib import Path

import pandas as pd
import streamlit as st

from components import format_money, normalise_id, role_color, role_label


@st.cache_data(ttl=20, show_spinner=False)
def load_edges(candidates: tuple[Path, ...]) -> tuple[pd.DataFrame, Path | None]:
    """Load a ready edge export once, retaining only fields needed for presentation."""
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".parquet":
                try:
                    loaded = pd.read_parquet(path, columns=["src", "dst", "sum_kzt"])
                except Exception:
                    loaded = pd.read_parquet(path, columns=["src", "dst"])
            else:
                loaded = pd.read_csv(
                    path,
                    encoding="utf-8-sig",
                    usecols=lambda column: column in {"src", "dst", "sum_kzt"},
                )
            if {"src", "dst"}.issubset(loaded.columns):
                loaded = loaded.copy()
                # These are lookup keys only. The UI never derives scores or graph metrics.
                loaded["_src_key"] = loaded["src"].map(normalise_id)
                loaded["_dst_key"] = loaded["dst"].map(normalise_id)
                return loaded, path
        except Exception:
            continue
    return pd.DataFrame(columns=["src", "dst"]), None


def _roles(nodes: pd.DataFrame, gids: set[str]) -> dict[str, object]:
    if not {"gid", "role"}.issubset(nodes.columns):
        return {}
    keys = nodes["_gid_key"] if "_gid_key" in nodes else nodes["gid"].map(normalise_id)
    subset = nodes.loc[keys.isin(gids), ["role"]]
    return dict(zip(keys.loc[subset.index], subset["role"]))


def _safe(value: object) -> str:
    return escape(str(value), quote=True)


def _network_canvas(selected_gid: object, incident: pd.DataFrame, nodes: pd.DataFrame, demo_mode: bool) -> str:
    selected = normalise_id(selected_gid)
    neighbors: list[str] = []
    for row in incident.itertuples(index=False):
        for gid in (normalise_id(row.src), normalise_id(row.dst)):
            if gid != selected and gid and gid not in neighbors:
                neighbors.append(gid)
    neighbors = neighbors[:10]
    roles = _roles(nodes, {selected, *neighbors})

    cx, cy = 510, 266
    position = {selected: (cx, cy)}
    radius_x, radius_y = 330, 175
    for index, gid in enumerate(neighbors):
        angle = (-pi / 2) + (2 * pi * index / max(1, len(neighbors)))
        position[gid] = (cx + radius_x * cos(angle), cy + radius_y * sin(angle))

    edge_svg: list[str] = []
    for index, row in enumerate(incident.itertuples(index=False)):
        src, dst = normalise_id(row.src), normalise_id(row.dst)
        if src not in position or dst not in position:
            continue
        x1, y1 = position[src]
        x2, y2 = position[dst]
        control_x = (x1 + x2) / 2 + (35 if y1 < y2 else -35)
        control_y = (y1 + y2) / 2 - 24
        amount = getattr(row, "sum_kzt", None)
        amount_label = format_money(amount) if amount is not None else "перевод"
        edge_svg.append(
            f'<g><path id="flow-{index}" class="flow-line" d="M {x1:.1f} {y1:.1f} Q {control_x:.1f} {control_y:.1f} {x2:.1f} {y2:.1f}" marker-end="url(#arrow)"/>'
            f'<circle r="3.5" class="flow-particle"><animateMotion dur="{2.4 + index * .18:.1f}s" repeatCount="indefinite" path="M {x1:.1f} {y1:.1f} Q {control_x:.1f} {control_y:.1f} {x2:.1f} {y2:.1f}"/></circle>'
            f'<title>{_safe(src)} -&gt; {_safe(dst)}: {_safe(amount_label)}</title></g>'
        )

    node_svg: list[str] = []
    for gid, (x, y) in position.items():
        is_selected = gid == selected
        role = roles.get(gid, "peripheral")
        color = role_color(role)
        label = "ВЫБРАН" if is_selected else role_label(role).upper()
        radius = 43 if is_selected else 29
        node_class = "network-node selected" if is_selected else "network-node"
        node_svg.append(
            f'<g class="{node_class}" transform="translate({x:.1f},{y:.1f})">'
            f'<circle class="node-halo" r="{radius + 13}" fill="{color}"/><circle class="node-core" r="{radius}" fill="#10192A" stroke="{color}" stroke-width="2"/>'
            f'<circle r="5" fill="{color}"/><text class="node-id" y="{radius + 18}">{_safe(gid)}</text><text class="node-role" y="{radius + 31}">{_safe(label)}</text></g>'
        )

    mode = "ДЕМО-СЕТЬ" if demo_mode else "НАБЛЮДАЕМЫЕ СВЯЗИ"
    return f"""
    <!doctype html><html><head><style>
      html,body {{ margin:0; background:transparent; overflow:hidden; font-family:Inter,Arial,sans-serif; }}
      .graph-shell {{ overflow:hidden; border:1px solid #283750; border-radius:18px; background:#0D1320; }}
      .graph-head {{ min-height:55px; display:flex; align-items:center; justify-content:space-between; gap:16px; padding:0 18px; border-bottom:1px solid #243149; background:linear-gradient(145deg,#111827,#0D1320); }}
      .graph-kicker {{ color:#7E8AA3; font-size:9px; font-weight:800; letter-spacing:1.6px; }}
      .graph-title {{ color:#F2F5FF; margin-top:4px; font-size:14px; font-weight:720; letter-spacing:-.15px; }}
      .graph-meta {{ color:#8190AC; font-size:9px; font-weight:800; letter-spacing:1px; white-space:nowrap; }}
      .canvas {{ height:520px; position:relative; overflow:hidden; background:radial-gradient(circle at 50% 45%,#172442 0,#0d1423 42%,#090e19 100%); }}
      .grid {{ position:absolute; inset:0; opacity:.27; background-image:linear-gradient(#43516d 1px,transparent 1px),linear-gradient(90deg,#43516d 1px,transparent 1px); background-size:38px 38px; mask-image:radial-gradient(ellipse at center,black,transparent 78%); }}
      .topline {{ position:absolute; z-index:2; left:18px; top:15px; color:#8493AE; font-size:10px; letter-spacing:1.8px; font-weight:800; }}
      .legend {{ position:absolute; z-index:2; right:18px; top:14px; color:#8392AB; font-size:10px; }}
      svg {{ position:absolute; inset:0; width:100%; height:100%; }}
      .flow-line {{ fill:none; stroke:#6F8DFF; stroke-width:1.4; opacity:.68; stroke-dasharray:4 7; animation:flow 2.8s linear infinite; }}
      .flow-particle {{ fill:#B6C8FF; filter:drop-shadow(0 0 7px #7a99ff); }}
      .node-halo {{ opacity:.10; animation:breathe 2.8s ease-in-out infinite; }}
      .node-core {{ filter:drop-shadow(0 0 12px #263e72); }}
      .selected .node-core {{ filter:drop-shadow(0 0 24px #728fff); }}
      .node-id {{ text-anchor:middle; fill:#F4F7FF; font-size:12px; font-weight:800; letter-spacing:.2px; }}
      .node-role {{ text-anchor:middle; fill:#8493AE; font-size:8px; letter-spacing:.8px; }}
      @keyframes flow {{ to {{ stroke-dashoffset:-44; }} }} @keyframes breathe {{ 50% {{ opacity:.22; transform:scale(1.06); transform-origin:center; }} }}
      @media (prefers-reduced-motion: reduce) {{ .flow-line,.node-halo {{ animation:none !important; }} .flow-particle {{ display:none; }} }}
      @media(max-width:680px) {{ .graph-meta {{ display:none; }} .legend {{ display:none; }} }}
    </style></head><body><div class="graph-shell"><div class="graph-head"><div><div class="graph-kicker">ДОКАЗАТЕЛЬСТВА СВЯЗЕЙ</div><div class="graph-title">Контур переводов · GID {_safe(selected)}</div></div><div class="graph-meta">НАПРАВЛЕНИЕ ВКЛЮЧЕНО</div></div><div class="canvas"><div class="grid"></div><div class="topline">{mode} &middot; {len(incident)} СВЯЗЕЙ</div><div class="legend">&#9679; направление потока &nbsp; &#9671; выбранный клиент</div>
    <svg viewBox="0 0 1020 520" preserveAspectRatio="xMidYMid meet" aria-label="Связи выбранного клиента"><defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#6F8DFF"/></marker></defs>{''.join(edge_svg)}{''.join(node_svg)}</svg></div></div></body></html>
    """


def _render_graph_empty(message: str) -> None:
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-head"><div><div class="panel-kicker">Доказательства связей</div><div class="panel-title">Окружение выбранного клиента</div></div><span class="quiet">Ожидание данных</span></div>
          <div class="panel-body"><div class="empty-state">{_safe(message)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_client_connections(selected_gid: object, edges: pd.DataFrame, nodes: pd.DataFrame, *, demo_mode: bool = False) -> None:
    """Render local incoming/outgoing evidence around the selected client."""
    if not {"src", "dst"}.issubset(edges.columns):
        _render_graph_empty("Связи появятся после загрузки готового файла с колонками src и dst.")
        return

    selected = normalise_id(selected_gid)
    if {"_src_key", "_dst_key"}.issubset(edges.columns):
        incident = edges.loc[(edges["_src_key"] == selected) | (edges["_dst_key"] == selected)].copy()
    else:
        incident = edges.loc[
            edges["src"].map(normalise_id).eq(selected) | edges["dst"].map(normalise_id).eq(selected)
        ].copy()
    if incident.empty:
        _render_graph_empty("Для выбранного клиента в доступном наборе нет наблюдаемых связей.")
        return

    if "sum_kzt" in incident.columns:
        incident["_sort"] = pd.to_numeric(incident["sum_kzt"], errors="coerce")
        incident = incident.sort_values("_sort", ascending=False, na_position="last")
    incident = incident.head(12)
    document = _network_canvas(selected_gid, incident, nodes, demo_mode)
    data_url = "data:text/html;charset=utf-8;base64," + b64encode(document.encode("utf-8")).decode("ascii")
    st.iframe(data_url, height=578)
