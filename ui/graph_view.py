"""Animated, read-only neighbourhood canvas for the selected customer."""

from __future__ import annotations

from base64 import b64encode
from html import escape
from math import cos, pi, sin, hypot, log1p, isfinite
from pathlib import Path

import pandas as pd
import streamlit as st

from components import format_money, normalise_id, open_selected_client, role_color, role_label


@st.cache_data(ttl=20, show_spinner=False)
def load_edges(candidates: tuple[Path, ...]) -> tuple[pd.DataFrame, Path | None]:
    """Load a ready edge export once, retaining only fields needed for presentation."""
    def unavailable(message: str):
        empty = pd.DataFrame(columns=["src", "dst"])
        empty.attrs["load_error"] = message
        return empty, None

    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".parquet":
                try:
                    loaded = pd.read_parquet(path)
                except Exception:
                    loaded = pd.read_parquet(path, columns=["src", "dst"])
            else:
                loaded = pd.read_csv(
                    path,
                    encoding="utf-8-sig",
                    usecols=lambda column: column in {"src", "dst", "sum_kzt", "n_tx"},
                    dtype={"src": "string", "dst": "string"},
                )
            if {"src", "dst"}.issubset(loaded.columns):
                loaded = loaded.copy()
                # Never send int64 IDs from parquet to JavaScript numeric cells.
                for column in ("src", "dst"):
                    loaded[column] = loaded[column].map(normalise_id).astype("string")
                # These are lookup keys only. The UI never derives scores or graph metrics.
                loaded["_src_key"] = loaded["src"].map(normalise_id)
                loaded["_dst_key"] = loaded["dst"].map(normalise_id)
                if loaded[["_src_key", "_dst_key"]].eq("").any().any():
                    return unavailable(f"{path.name}: пустой GID в связях. Исправьте выгрузку и обновите данные.")
                if "sum_kzt" in loaded:
                    loaded["sum_kzt"] = pd.to_numeric(loaded["sum_kzt"], errors="coerce")
                    if not loaded.sum_kzt.map(lambda value: pd.notna(value) and isfinite(value) and value >= 0).all():
                        return unavailable(f"{path.name}: некорректная сумма перевода. Исправьте выгрузку и обновите данные.")
                return loaded, path
            return unavailable(f"{path.name}: нужны колонки src и dst. Связи не загружены.")
        except Exception:
            return unavailable(f"Не удалось прочитать {path.name}. Связи недоступны; проверьте файл и обновите данные.")
    return unavailable("Файл связей не найден. Нельзя определить, есть ли у клиента переводы.")


def _roles(nodes: pd.DataFrame, gids: set[str]) -> dict[str, object]:
    if not {"gid", "role"}.issubset(nodes.columns):
        return {}
    keys = nodes["_gid_key"] if "_gid_key" in nodes else nodes["gid"].map(normalise_id)
    subset = nodes.loc[keys.isin(gids), ["role"]]
    return dict(zip(keys.loc[subset.index], subset["role"]))


def _safe(value: object) -> str:
    return escape(str(value), quote=True)


def _network_canvas(selected_gid: object, incident: pd.DataFrame, nodes: pd.DataFrame, demo_mode: bool, cluster_colors: bool = False) -> str:
    selected = normalise_id(selected_gid)
    neighbors: list[str] = []
    for row in incident.itertuples(index=False):
        for gid in (normalise_id(row.src), normalise_id(row.dst)):
            if gid != selected and gid and gid not in neighbors:
                neighbors.append(gid)
    neighbors = neighbors[:10]
    roles = _roles(nodes, {selected, *neighbors})
    cluster_map = dict(zip(nodes.gid.map(normalise_id), nodes.cluster_id.map(normalise_id))) if "cluster_id" in nodes else {}
    cluster_palette = {cluster: f"hsl({index * 137.508 % 360:.1f} 70% 65%)"
                       for index, cluster in enumerate(sorted(set(cluster_map.values()) - {"—", ""}))}

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
        distance = hypot(x2 - x1, y2 - y1)
        if distance:
            dx, dy = (x2 - x1) / distance, (y2 - y1) / distance
            start_radius = 47 if src == selected else 33
            end_radius = 51 if dst == selected else 37
            x1, y1 = x1 + dx * start_radius, y1 + dy * start_radius
            x2, y2 = x2 - dx * end_radius, y2 - dy * end_radius
        control_x = (x1 + x2) / 2 + (35 if y1 < y2 else -35)
        control_y = (y1 + y2) / 2 - 24
        amount = getattr(row, "sum_kzt", None)
        amount_label = format_money(amount) if amount is not None else "перевод"
        path = f"M {x1:.1f} {y1:.1f} Q {control_x:.1f} {control_y:.1f} {x2:.1f} {y2:.1f}"
        if src == dst:
            path = f"M {x1 - 25:.1f} {y1 - 30:.1f} C {x1 - 100:.1f} {y1 - 120:.1f} {x1 + 100:.1f} {y1 - 120:.1f} {x1 + 30:.1f} {y1 - 35:.1f}"
        width = 1.3 + min(2.0, log1p(float(amount or 0)) / 8)
        direction = "outgoing" if src == selected else "incoming"
        color = "#FFB270" if direction == "outgoing" else "#58CFFF"
        edge_svg.append(
            f'<g><path id="flow-{index}" class="flow-line" style="stroke:{color};stroke-width:{width:.1f}" d="{path}" marker-end="url(#arrow-{direction})"/>'
            f'<circle r="3.5" class="flow-particle" style="fill:{color}"><animateMotion dur="{2.4 + index * .18:.1f}s" repeatCount="indefinite" path="{path}"/></circle>'
            f'<title>{_safe(src)} -&gt; {_safe(dst)}: {_safe(amount_label)}</title></g>'
        )

    node_svg: list[str] = []
    for gid, (x, y) in position.items():
        is_selected = gid == selected
        role = roles.get(gid, "unassigned")
        color = role_color(role)
        if cluster_colors:
            cluster = cluster_map.get(gid, "—")
            color = cluster_palette.get(cluster, "#8090AB")
        label = "ВЫБРАН" if is_selected else role_label(role).upper()
        if cluster_colors and not is_selected:
            label = f"КЛАСТЕР {cluster_map.get(gid, '—')}"
        radius = 43 if is_selected else 29
        node_class = "network-node selected" if is_selected else "network-node"
        node_svg.append(
            f'<g class="{node_class}" transform="translate({x:.1f},{y:.1f})">'
            f'<title>GID {_safe(gid)} · {_safe(role_label(role))} · кластер {_safe(cluster_map.get(gid, "—"))}</title>'
            f'<circle class="node-halo" r="{radius + 13}" fill="{color}"/><circle class="node-core" r="{radius}" fill="#10192A" stroke="{color}" stroke-width="2"/>'
            f'<circle r="5" fill="{color}"/><text class="node-id" y="{radius + 18}">{_safe(gid)}</text><text class="node-role" y="{radius + 31}">{_safe(label)}</text></g>'
        )

    mode = "ДЕМО-СЕТЬ" if demo_mode else "НАБЛЮДАЕМЫЕ СВЯЗИ"
    document = f"""
    <!doctype html><html><head><style>
      html,body {{ margin:0; background:transparent; overflow:hidden; font-family:Inter,Arial,sans-serif; }}
      .graph-shell {{ overflow:hidden; border:1px solid #283750; border-radius:18px; background:#0D1320; }}
      .graph-head {{ min-height:55px; display:flex; align-items:center; justify-content:space-between; gap:16px; padding:0 18px; border-bottom:1px solid #243149; background:linear-gradient(145deg,#111827,#0D1320); }}
      .graph-kicker {{ color:#7E8AA3; font-size:9px; font-weight:800; letter-spacing:1.6px; }}
      .graph-title {{ color:#F2F5FF; margin-top:4px; font-size:14px; font-weight:720; letter-spacing:-.15px; }}
      .graph-meta {{ color:#8190AC; font-size:9px; font-weight:800; letter-spacing:1px; white-space:nowrap; }}
      .canvas {{ height:420px; position:relative; overflow:hidden; background:radial-gradient(circle at 50% 45%,#172442 0,#0d1423 42%,#090e19 100%); }}
      .grid {{ position:absolute; inset:0; opacity:.27; background-image:linear-gradient(#43516d 1px,transparent 1px),linear-gradient(90deg,#43516d 1px,transparent 1px); background-size:38px 38px; mask-image:radial-gradient(ellipse at center,black,transparent 78%); }}
      .topline {{ position:absolute; z-index:2; left:18px; top:15px; color:#8493AE; font-size:10px; letter-spacing:1.8px; font-weight:800; }}
      .legend {{ position:absolute; z-index:2; left:18px; bottom:18px; color:#B9C6DF; font-size:11px; }}
      .zoom-tools {{ position:absolute; z-index:3; bottom:12px; right:14px; display:flex; gap:6px; }}
      button {{ background:#19253c; border:1px solid #3c5076; color:#dce6ff; border-radius:7px; padding:7px 12px; cursor:pointer; }}
      button:focus-visible {{ outline:2px solid #9cb0ff; }}
      svg {{ touch-action:none; cursor:grab; }}
      svg {{ position:absolute; inset:0; width:100%; height:100%; }}
      .flow-line {{ fill:none; stroke:#6F8DFF; stroke-width:1.4; opacity:.68; stroke-dasharray:4 7; animation:flow 2.8s linear infinite; }}
      .flow-particle {{ fill:#B6C8FF; filter:drop-shadow(0 0 7px #7a99ff); }}
      .node-halo {{ opacity:.10; }}
      .node-core {{ filter:drop-shadow(0 0 12px #263e72); }}
      .selected .node-core {{ filter:drop-shadow(0 0 24px #728fff); }}
      .node-id {{ text-anchor:middle; fill:#F4F7FF; font-size:17px; font-weight:700; }}
      .node-role {{ text-anchor:middle; fill:#A7B6D0; font-size:10px; letter-spacing:.6px; }}
      @keyframes flow {{ to {{ stroke-dashoffset:-44; }} }} @keyframes breathe {{ 50% {{ opacity:.22; transform:scale(1.06); transform-origin:center; }} }}
      @media (prefers-reduced-motion: reduce) {{ .flow-line,.node-halo {{ animation:none !important; }} .flow-particle {{ display:none; }} }}
      @media(max-width:680px) {{ .graph-meta {{ display:none; }} }}
    </style></head><body><div class="graph-shell"><div class="graph-head"><div><div class="graph-kicker">НАБЛЮДАЕМЫЕ ПЕРЕВОДЫ</div><div class="graph-title">Контур переводов · GID {_safe(selected)}</div></div><div class="graph-meta">НАПРАВЛЕНИЕ ВКЛЮЧЕНО</div></div><div class="canvas"><div class="grid"></div><div class="topline">{mode} &middot; {len(incident)} СВЯЗЕЙ</div><div class="legend"><span style="color:#58CFFF">→ Входящие</span> &nbsp; <span style="color:#FFB270">→ Исходящие</span></div>
    <svg id="network" viewBox="0 0 1020 520" preserveAspectRatio="xMidYMid meet" aria-label="Связи выбранного клиента"><defs><marker id="arrow-incoming" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#58CFFF"/></marker><marker id="arrow-outgoing" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M 0 0 L 10 5 L 0 10 z" fill="#FFB270"/></marker></defs>{''.join(edge_svg)}{''.join(node_svg)}</svg><div class="zoom-tools"><button id="zoom-in" aria-label="Приблизить">+</button><button id="zoom-out" aria-label="Отдалить">−</button><button id="zoom-reset">Сброс</button></div></div></div>
    """
    return document + """<script>
    const svg = document.getElementById('network');
    let box = [0, 0, 1020, 520], start;
    const draw = () => svg.setAttribute('viewBox', box.join(' '));
    const zoom = factor => {
      const width = Math.max(255, Math.min(2040, box[2] * factor));
      const height = width * 520 / 1020;
      box = [box[0] + (box[2]-width)/2, box[1] + (box[3]-height)/2, width, height]; draw();
    };
    document.getElementById('zoom-in').onclick = () => zoom(.8);
    document.getElementById('zoom-out').onclick = () => zoom(1.25);
    document.getElementById('zoom-reset').onclick = () => {box = [0,0,1020,520]; draw();};
    svg.onpointerdown = e => {start = [e.clientX,e.clientY,...box]; svg.setPointerCapture(e.pointerId);};
    svg.onpointermove = e => {if(!start) return;
      const scale = Math.max(start[4]/svg.clientWidth, start[5]/svg.clientHeight);
      box[0] = start[2]-(e.clientX-start[0])*scale;
      box[1] = start[3]-(e.clientY-start[1])*scale; draw();};
    svg.onpointerup = svg.onpointercancel = () => {start = null;};
    </script></body></html>"""


def _render_graph_empty(message: str, status: str = "Связи недоступны") -> None:
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-head"><div><div class="panel-kicker">Наблюдаемые переводы</div><div class="panel-title">Окружение выбранного клиента</div></div><span class="quiet">{_safe(status)}</span></div>
          <div class="panel-body"><div class="empty-state">{_safe(message)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_client_connections(selected_gid: object, edges: pd.DataFrame, nodes: pd.DataFrame, *, demo_mode: bool = False) -> None:
    """Render local incoming/outgoing evidence around the selected client."""
    if edges.attrs.get("load_error"):
        _render_graph_empty("Связи недоступны. Проверьте сообщение о загрузке; отсутствие графа не означает отсутствие переводов.")
        return
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
        _render_graph_empty("Для выбранного клиента в доступном наборе нет наблюдаемых связей.", "Нет наблюдаемых связей")
        return

    if "sum_kzt" in incident.columns:
        incident["_sort"] = pd.to_numeric(incident["sum_kzt"], errors="coerce")
        incident = incident.sort_values("_sort", ascending=False, na_position="last")
    total = len(incident)
    direction = st.radio("Направление связей", ["Все", "Входящие", "Исходящие"], horizontal=True, key=f"direction_{selected}")
    if direction == "Входящие":
        incident = incident.loc[incident.dst.map(normalise_id).eq(selected)]
    elif direction == "Исходящие":
        incident = incident.loc[incident.src.map(normalise_id).eq(selected)]
    filtered_total = len(incident)
    if incident.empty:
        st.info(f"В выбранном направлении нет наблюдаемых переводов. Всего у GID {selected}: {total} связей.")
        return
    all_incident = incident
    peers = []
    for row in incident.itertuples():
        peer = normalise_id(row.dst) if normalise_id(row.src) == selected else normalise_id(row.src)
        if peer != selected and peer not in peers:
            peers.append(peer)
    page_size = 10
    page_count = max(1, (len(peers) + page_size - 1) // page_size)
    controls = st.columns(2)
    with controls[0]:
        color_mode = st.radio("Подсветка", ["Роли", "Кластеры"], horizontal=True)
    with controls[1]:
        page_key = f"graph_page_{selected}_{direction}"
        if st.session_state.get(page_key, 0) >= page_count:
            st.session_state[page_key] = 0
        page = st.selectbox("Группа связей", range(page_count), format_func=lambda p: f"{p + 1} / {page_count}", key=page_key)
    shown = {selected, *peers[page * page_size:(page + 1) * page_size]}
    incident = all_incident.loc[all_incident.src.map(normalise_id).isin(shown) & all_incident.dst.map(normalise_id).isin(shown)]
    document = _network_canvas(selected_gid, incident, nodes, demo_mode, cluster_colors=color_mode == "Кластеры")
    data_url = "data:text/html;charset=utf-8;base64," + b64encode(document.encode("utf-8")).decode("ascii")
    st.iframe(data_url, height=478)
    st.caption(f"Показано {len(incident)} из {filtered_total} связей в фильтре · всего у клиента {total}. До 10 соседей на странице, по сумме переводов. Стрелка указывает получателя. Схему можно перемещать и масштабировать.")
    available = set(nodes.gid.map(normalise_id))
    known_peers = [gid for gid in peers if gid in available]
    if known_peers:
        picker, action = st.columns([3, 2], vertical_alignment="bottom")
        with picker:
            peer_gid = st.selectbox("Связанный GID", known_peers, key=f"graph_peer_{selected}_{direction}")
        with action:
            st.button("Продолжить по связи →", on_click=open_selected_client, args=(f"graph_peer_{selected}_{direction}",), width="stretch")
