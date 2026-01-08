#!/usr/bin/env python3
"""Render an interactive verification network linking pathway themes to enhancer cell types."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import json
import math

DecisionPalette = {
    "Supported": "#2ca02c",
    "Partial": "#ff7f0e",
    "Hypothesis": "#1f77b4",
    "Refuted": "#d62728",
}
ThemeColor = "#6a51a3"
CellColor = "#17becf"
CriticDash = {
    "Contextualize": "8,4",
    "Discard": "3,6",
}
CriticColorOverride = {
    "Discard": "#9e9e9e",
}
CriticWidth = {
    "Contextualize": 2.5,
    "Discard": 1.8,
}


def parse_table(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("-") or line.startswith("theme"):
                continue
            parts = [segment.strip() for segment in line.split("|")]
            if len(parts) < 7:
                continue
            rows.append(
                {
                    "theme": parts[0],
                    "direction": parts[1],
                    "cell_type": parts[2],
                    "decision": parts[3],
                    "effect_size": parts[4],
                    "q_value": parts[5],
                    "reason": parts[6],
                }
            )
    return rows


def _unique_preserve(sequence: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in sequence:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _compute_positions(rows: Sequence[Dict[str, str]], width: int = 1200, height: int = 800) -> Dict[str, Dict[str, float]]:
    theme_keys = _unique_preserve(f"{row['theme']} ({row['direction']})" for row in rows)
    theme_positions: Dict[str, Dict[str, float]] = {}
    count = max(len(theme_keys), 1)
    center_y = height * 0.5
    for idx, theme_key in enumerate(theme_keys):
        x = width * (idx + 1) / (count + 1)
        theme_positions[theme_key] = {"x": x, "y": center_y}

    theme_to_cells: Dict[str, List[str]] = {}
    for row in rows:
        theme_key = f"{row['theme']} ({row['direction']})"
        theme_to_cells.setdefault(theme_key, []).append(row["cell_type"])

    cell_positions: Dict[str, Dict[str, float]] = {}
    for theme_key, cells in theme_to_cells.items():
        if not cells:
            continue
        hub = theme_positions[theme_key]
        unique_cells = _unique_preserve(cells)
        total = len(unique_cells)
        if total == 1:
            angle_step = 0
        else:
            angle_step = 2 * math.pi / total
        radius = 160
        for idx, cell in enumerate(unique_cells):
            if cell in cell_positions:
                continue
            angle = angle_step * idx
            cx = hub["x"] + radius * math.cos(angle)
            cy = hub["y"] + radius * math.sin(angle)
            cx = max(80, min(width - 80, cx))
            cy = max(80, min(height - 80, cy))
            cell_positions[cell] = {"x": cx, "y": cy}

    positions = {**{theme: {"x": pos["x"], "y": pos["y"], "type": "theme"} for theme, pos in theme_positions.items()}}
    for cell, pos in cell_positions.items():
        positions[cell] = {"x": pos["x"], "y": pos["y"], "type": "cell"}
    return positions


def build_html(rows: Sequence[Dict[str, str]], title: str) -> str:
    width, height = 1200, 800
    positions = _compute_positions(rows, width, height)

    nodes_payload = []
    theme_counts: Dict[str, int] = {}
    cell_counts: Dict[str, int] = {}
    for row in rows:
        theme_key = f"{row['theme']} ({row['direction']})"
        theme_counts[theme_key] = theme_counts.get(theme_key, 0) + 1
        cell_counts[row["cell_type"]] = cell_counts.get(row["cell_type"], 0) + 1

    for node_id, meta in positions.items():
        if meta["type"] == "theme":
            count = theme_counts.get(node_id, 1)
            nodes_payload.append(
                {
                    "id": node_id,
                    "label": node_id,
                    "class": "theme",
                    "x": meta["x"],
                    "y": meta["y"],
                    "radius": 32 + count * 2,
                    "color": ThemeColor,
                    "fontSize": 18,
                }
            )
        else:
            count = cell_counts.get(node_id, 1)
            nodes_payload.append(
                {
                    "id": node_id,
                    "label": node_id,
                    "class": "cell",
                    "x": meta["x"],
                    "y": meta["y"],
                    "radius": 18 + count * 1.2,
                    "color": CellColor,
                    "fontSize": 14,
                }
            )

    edges_payload = []
    for idx, row in enumerate(rows):
        theme_key = f"{row['theme']} ({row['direction']})"
        critic_action = row.get("critic_action")
        critic_concern = row.get("critic_concern")
        critic_recommendation = row.get("critic_recommendation")
        edges_payload.append(
            {
                "id": f"edge_{idx}",
                "source": theme_key,
                "target": row["cell_type"],
                "color": CriticColorOverride.get(critic_action, DecisionPalette.get(row["decision"], "#7f7f7f")),
                "tooltip": (
                    f"{row['theme']} ({row['direction']}) → {row['cell_type']}\\n"
                    f"Decision: {row['decision']} | Effect {row['effect_size']} | q={row['q_value']}\\n"
                    f"{row['reason']}"
                    + (
                        f"\\nCritic: {critic_action} — {critic_concern}"
                        + (f" ({critic_recommendation})" if critic_recommendation else "")
                        if critic_action
                        else ""
                    )
                ),
                "dash": CriticDash.get(critic_action),
                "width": CriticWidth.get(critic_action, 2.5),
            }
        )

    legend_items = [{"label": label, "color": color} for label, color in DecisionPalette.items()]

    template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: #fafafa;
      color: #222;
    }}
    header {{
      padding: 16px 24px;
      background: #ffffff;
      box-shadow: 0 2px 6px rgba(0,0,0,0.1);
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    #network-container {{
      position: relative;
      padding: 16px;
    }}
    svg {{
      background: #ffffff;
      border-radius: 12px;
      box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }}
    .legend {{
      position: absolute;
      top: 32px;
      left: 32px;
      background: rgba(255, 255, 255, 0.92);
      border-radius: 8px;
      padding: 12px 16px;
      box-shadow: 0 4px 12px rgba(0,0,0,0.12);
      font-size: 14px;
      line-height: 1.6;
    }}
    .legend-title {{
      font-weight: 600;
      margin-bottom: 6px;
    }}
    .legend-item {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .legend-swatch {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
    }}
    .node-label {{
      pointer-events: none;
      paint-order: stroke;
      stroke: #fff;
      stroke-width: 3px;
      alignment-baseline: middle;
    }}
    .theme-label {{
      font-size: 18px;
      font-weight: 600;
      fill: #2f1f53;
    }}
    .cell-label {{
      font-size: 14px;
      fill: #0d3b4c;
    }}
    .tooltip {{
      position: absolute;
      pointer-events: none;
      background: rgba(33, 33, 33, 0.92);
      color: #fff;
      padding: 8px 10px;
      border-radius: 6px;
      font-size: 13px;
      max-width: 320px;
      display: none;
      white-space: pre-wrap;
    }}
  </style>
</head>
<body>
  <header>
    <h1 style="margin:0;font-size:22px;">{title}</h1>
    <p style="margin:4px 0 0 0;font-size:15px;color:#555;">
      Drag nodes to refine layout. Hover connections for effect sizes and rationale.
    </p>
  </header>
  <div id="network-container">
    <div class="legend">
      <div class="legend-title">Decision</div>
      {''.join(f'<div class="legend-item"><div class="legend-swatch" style="background:{item["color"]};"></div>{item["label"]}</div>' for item in legend_items)}
    </div>
    <div id="tooltip" class="tooltip"></div>
    <svg id="network" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
      <g id="edges"></g>
      <g id="nodes"></g>
    </svg>
  </div>
  <script>
    const nodes = {json.dumps(nodes_payload)};
    const edges = {json.dumps(edges_payload)};
    const positions = {json.dumps(positions)};
    const svg = document.getElementById("network");
    const edgeLayer = document.getElementById("edges");
    const nodeLayer = document.getElementById("nodes");
    const tooltip = document.getElementById("tooltip");

    const nodeElements = new Map();
    const labelElements = new Map();
    const edgeElements = new Map();

    edges.forEach(edge => {{
      const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
      line.setAttribute("stroke", edge.color);
      line.setAttribute("stroke-width", edge.width || 2.5);
      if (edge.dash) {{
        line.setAttribute("stroke-dasharray", edge.dash);
      }}
      line.dataset.source = edge.source;
      line.dataset.target = edge.target;
      line.dataset.tooltip = edge.tooltip;
      edgeLayer.appendChild(line);
      edgeElements.set(edge.id, line);
    }});

    nodes.forEach(node => {{
      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("r", node.radius);
      circle.setAttribute("fill", node.color);
      circle.dataset.id = node.id;
      circle.style.cursor = "grab";
      nodeLayer.appendChild(circle);
      nodeElements.set(node.id, circle);

      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.textContent = node.label;
      text.setAttribute("class", `node-label ${{node.class}}-label`);
      text.dataset.id = node.id;
      text.setAttribute("text-anchor", "middle");
      nodeLayer.appendChild(text);
      labelElements.set(node.id, text);
    }});

    function updateLayout() {{
      nodes.forEach(node => {{
        const pos = positions[node.id];
        const circle = nodeElements.get(node.id);
        const label = labelElements.get(node.id);
        circle.setAttribute("cx", pos.x);
        circle.setAttribute("cy", pos.y);
        label.setAttribute("x", pos.x);
        label.setAttribute("y", pos.y + (node.class === "theme" ? -(node.radius + 10) : node.radius + 18));
      }});
      edges.forEach(edge => {{
        const source = positions[edge.source];
        const target = positions[edge.target];
        const line = edgeElements.get(edge.id);
        line.setAttribute("x1", source.x);
        line.setAttribute("y1", source.y);
        line.setAttribute("x2", target.x);
        line.setAttribute("y2", target.y);
      }});
    }}
    updateLayout();

    let activeNode = null;
    let dragOffset = {{x: 0, y: 0}};

    function getMousePosition(evt) {{
      const rect = svg.getBoundingClientRect();
      return {{
        x: (evt.clientX - rect.left),
        y: (evt.clientY - rect.top)
      }};
    }}

    function handlePointerDown(evt) {{
      const nodeId = evt.target.dataset.id;
      if (!nodeId) {{
        return;
      }}
      svg.setPointerCapture(evt.pointerId);
      activeNode = nodeId;
      const pos = positions[nodeId];
      const pointer = getMousePosition(evt);
      dragOffset = {{
        x: pointer.x - pos.x,
        y: pointer.y - pos.y
      }};
      evt.target.style.cursor = "grabbing";
    }}

    function handlePointerMove(evt) {{
      if (!activeNode) {{
        const edge = evt.target;
        if (edge.dataset && edge.dataset.tooltip) {{
          tooltip.style.display = "block";
          tooltip.innerText = edge.dataset.tooltip;
          tooltip.style.left = `${{evt.pageX + 12}}px`;
          tooltip.style.top = `${{evt.pageY + 12}}px`;
        }} else {{
          tooltip.style.display = "none";
        }}
        return;
      }}
      const pointer = getMousePosition(evt);
      positions[activeNode].x = Math.max(40, Math.min({width - 40}, pointer.x - dragOffset.x));
      positions[activeNode].y = Math.max(40, Math.min({height - 40}, pointer.y - dragOffset.y));
      updateLayout();
    }}

    function handlePointerUp(evt) {{
      if (activeNode) {{
        const circle = nodeElements.get(activeNode);
        circle.style.cursor = "grab";
      }}
      activeNode = null;
      svg.releasePointerCapture(evt.pointerId);
    }}

    svg.addEventListener("pointerdown", handlePointerDown);
    svg.addEventListener("pointermove", handlePointerMove);
    svg.addEventListener("pointerup", handlePointerUp);
    svg.addEventListener("pointerleave", () => {{
      tooltip.style.display = "none";
    }});

    svg.addEventListener("wheel", evt => {{
      evt.preventDefault();
      const scale = evt.deltaY < 0 ? 1.05 : 0.95;
      const viewBox = svg.viewBox.baseVal;
      const mx = evt.offsetX / svg.clientWidth;
      const my = evt.offsetY / svg.clientHeight;
      const newWidth = viewBox.width * scale;
      const newHeight = viewBox.height * scale;
      viewBox.x += (viewBox.width - newWidth) * mx;
      viewBox.y += (viewBox.height - newHeight) * my;
      viewBox.width = Math.min(Math.max(newWidth, 600), {width});
      viewBox.height = Math.min(Math.max(newHeight, 400), {height});
    }});
  </script>
</body>
</html>
"""
    return template
def build_network(rows: Sequence[Dict[str, str]], title: str) -> Network:
    net = Network(height="720px", width="100%", bgcolor="#ffffff", font_color="#1f1f1f", notebook=False)
    net.force_atlas_2based(central_gravity=0.018, gravity=-48, spring_length=160, spring_strength=0.005, damping=0.85)
    net.toggle_physics(True)
    net.set_options(
        """
    var options = {
      interaction: { hover: true, dragNodes: true, dragView: true, zoomView: true },
      physics: {
        stabilization: { iterations: 120, fit: true },
        maxVelocity: 30,
        timestep: 0.6
      }
    }
    """
    )
    net.heading = title

    # collect nodes
    theme_counts: Dict[str, int] = {}
    cell_counts: Dict[str, int] = {}
    for row in rows:
        theme_key = f"{row['theme']} ({row['direction']})"
        theme_counts[theme_key] = theme_counts.get(theme_key, 0) + 1
        cell_counts[row["cell_type"]] = cell_counts.get(row["cell_type"], 0) + 1

    for theme_key, count in theme_counts.items():
        direction = theme_key.split("(")[-1].rstrip(")")
        net.add_node(
            theme_key,
            label=theme_key,
            title=f"{theme_key}<br>Connections: {count}",
            color=ThemeColor,
            shape="ellipse",
            size=26 + count * 2,
            mass=3.0,
            font={"size": 18},
        )

    for cell_type, count in cell_counts.items():
        net.add_node(
            cell_type,
            label=cell_type,
            title=f"{cell_type}<br>Supports: {count}",
            color=CellColor,
            shape="dot",
            size=14 + count * 1.2,
            mass=1.0,
            font={"size": 14},
        )

    for row in rows:
        theme_key = f"{row['theme']} ({row['direction']})"
        color = DecisionPalette.get(row["decision"], "#7f7f7f")
        tooltip = (
            f"<b>{row['theme']} ({row['direction']}) → {row['cell_type']}</b><br>"
            f"Decision: {row['decision']}<br>"
            f"Effect: {row['effect_size']} | q={row['q_value']}<br>"
            f"{row['reason']}"
        )
        net.add_edge(
            theme_key,
            row["cell_type"],
            color=color,
            title=tooltip,
            width=2.0,
            smooth=False,
        )

    # Legend nodes (fixed position, no physics)
    x0, y0 = -400, -400
    for idx, (label, color) in enumerate(DecisionPalette.items()):
        legend_id = f"legend_{label}"
        net.add_node(
            legend_id,
            label=label,
            color=color,
            shape="box",
            size=16,
            font={"size": 14},
            x=x0,
            y=y0 - idx * 60,
            physics=False,
            fixed={"x": True, "y": True},
        )

    return net


def render_verification_network(rows: Sequence[Dict[str, str]], output: Path, title: str) -> Path:
    """Render verification rows to an interactive HTML network."""
    if not rows:
        raise ValueError("No verification records to visualize.")
    output.parent.mkdir(parents=True, exist_ok=True)
    html = build_html(rows, title)
    output.write_text(html, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Render verification network as interactive HTML")
    parser.add_argument("verification_table", type=Path, help="Path to verification_table.txt")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/figures/verification_network.html"),
        help="Output HTML path for the interactive network",
    )
    args = parser.parse_args()

    rows = parse_table(args.verification_table)
    if not rows:
        raise SystemExit("No data parsed from verification table")

    title = f"Verification network — {args.verification_table.name}"
    render_verification_network(rows, args.output, title=title)
    print(f"Wrote network to {args.output}")


if __name__ == "__main__":
    main()
