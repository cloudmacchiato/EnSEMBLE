"""Generate publication-ready figures and a PDF report from agent outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from .md_to_pdf import md_to_pdf

THEME_COLORS = {"UP": "#d62728", "DOWN": "#1f77b4"}
HELPER_COLORS = {"celltype": "#8c564b", "tf_family": "#2ca02c", "theme_only": "#7f7f7f"}


def load_json(path: Path):
  with path.open("r", encoding="utf-8") as handle:
    return json.load(handle)


def ensure_dir(path: Path) -> Path:
  path.mkdir(parents=True, exist_ok=True)
  return path


def plot_theme_helper_links(output_dir: Path,
                            themes: Dict[str, List[dict]],
                            claims: List[dict],
                            helpers_meta: List[dict],
                            claim_reviews: List[dict]) -> tuple[Path, Path, str]:
  fig_dir = ensure_dir(output_dir / "figures")
  base = fig_dir / "figure01_theme_helper_network"
  pdf_path = base.with_suffix(".pdf")
  png_path = base.with_suffix(".png")

  supported_keys = {
      (review.get("theme_id"), review.get("helper_name"))
      for review in claim_reviews
      if review.get("verdict") == "Supported"
  }
  theme_nodes: List[Tuple[str, str, str]] = []
  for direction in ("UP", "DOWN"):
    for theme in themes.get(direction, []):
      has_supported = any(
          key for key in supported_keys
          if key[0] == theme["theme_id"]
      )
      if has_supported:
        theme_nodes.append((theme["theme_id"], direction, theme["label"]))

  if not theme_nodes:
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, "No themes to display.", ha="center", va="center")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path, "Figure 1. Theme ↔ Helper Links"

  theme_positions = {theme_id: i for i, (theme_id, _, _) in enumerate(theme_nodes)}

  helper_names = []
  helper_classes = {}
  for claim in claims:
    name = claim.get("helper_name")
    if name and name not in helper_names:
      helper_names.append(name)
      helper_classes[name] = claim.get("helper_class", "celltype")

  helper_positions = {name: idx for idx, name in enumerate(helper_names)}
  verdict_lookup = {
      (review.get("theme_id"), review.get("helper_name")): review.get("verdict")
      for review in claim_reviews
  }

  helper_lookup = {
      helper["helper_name"]: helper
      for helper in helpers_meta
      if isinstance(helper.get("helper_name"), str)
  }

  fig, ax = plt.subplots(figsize=(8.5, max(6, len(theme_nodes) * 0.6)))

  for theme_id, direction, label in theme_nodes:
    y = theme_positions[theme_id]
    ax.scatter(0, y, color=THEME_COLORS.get(direction, "#333333"), s=120, edgecolor="black", zorder=3)
    ax.text(-0.02, y, label, va="center", ha="right", fontsize=10, color="#1a1a1a")

  for helper_name, y in helper_positions.items():
    helper_class = helper_classes.get(helper_name, "celltype")
    ax.scatter(1, y, color=HELPER_COLORS.get(helper_class, "#7f7f7f"), s=100, edgecolor="black", zorder=3)
    ax.text(1.02, y, helper_name, va="center", ha="left", fontsize=9)

  for claim in claims:
    helper_name = claim.get("helper_name")
    if not helper_name or helper_name not in helper_positions:
      continue
    theme_id = claim["theme_id"]
    if theme_id not in theme_positions:
      continue
    y_theme = theme_positions[theme_id]
    y_helper = helper_positions[helper_name]
    helper_meta = helper_lookup.get(helper_name, {})
    verdict = verdict_lookup.get((theme_id, helper_name))
    if verdict:
      linewidth = {
          "Supported": 4.0,
          "Partial": 2.6,
          "Hypothesis": 1.8,
          "Not supported": 1.2,
      }.get(verdict, 2.0)
    else:
      q_val = helper_meta.get("q_value")
      if q_val is not None:
        linewidth = min(4.0, max(1.2, -math.log10(max(q_val, 1e-6))))
      else:
        confidence = claim.get("confidence", "medium")
        linewidth = {"high": 3.0, "medium": 2.0, "low": 1.2}.get(confidence, 1.5)
    ax.plot([0, 1], [y_theme, y_helper], color="#bbbbbb", linewidth=linewidth, alpha=0.85, zorder=1)

  ax.set_xlim(-0.3, 1.3)
  ax.set_ylim(-1, max(len(theme_nodes), len(helper_positions)) + 0.5)
  ax.axis("off")
  ax.set_title("Figure 1. Theme ↔ Helper Links (direction vs program class)", fontsize=14, pad=20)

  theme_handles = [
      Patch(facecolor=THEME_COLORS["UP"], edgecolor="black", label="Themes (UP)"),
      Patch(facecolor=THEME_COLORS["DOWN"], edgecolor="black", label="Themes (DOWN)"),
  ]
  helper_handles = [
      Patch(facecolor=HELPER_COLORS["celltype"], edgecolor="black", label="Helpers: cell-type"),
      Patch(facecolor=HELPER_COLORS["tf_family"], edgecolor="black", label="Helpers: TF-family"),
      Patch(facecolor=HELPER_COLORS["theme_only"], edgecolor="black", label="Theme-only hypothesis"),
  ]
  handles = theme_handles + helper_handles
  ax.legend(handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=3,
            frameon=False,
            fontsize=9)

  fig.tight_layout()
  fig.savefig(pdf_path, dpi=300)
  fig.savefig(png_path, dpi=300)
  plt.close(fig)
  return pdf_path, png_path, "Figure 1. Theme ↔ Helper Links"


def plot_theme_summary(output_dir: Path, themes: Dict[str, List[dict]], claim_reviews: List[dict]) -> tuple[Path, Path, str]:
  fig_dir = ensure_dir(output_dir / "figures")
  base = fig_dir / "figure02_theme_summary"
  pdf_path = base.with_suffix(".pdf")
  png_path = base.with_suffix(".png")
  supported_theme_ids = {
      review.get("theme_id")
      for review in claim_reviews
      if review.get("verdict") == "Supported"
  }
  records = []
  for direction, theme_list in themes.items():
    for theme in theme_list:
      if theme["theme_id"] not in supported_theme_ids:
        continue
      records.append({
          "Theme": theme["label"],
          "Direction": direction,
          "Effect": theme["effect"],
          "q_value": theme["q_value"],
      })
  if not records:
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, "No condensed themes available.", ha="center", va="center")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path, "Figure 2. Condensed Themes Overview"
  df = pd.DataFrame(records)
  df.sort_values(["Direction", "Effect"], ascending=[True, False], inplace=True)

  fig, ax = plt.subplots(figsize=(8, max(5, len(df) * 0.35)))
  colors = df["Direction"].map(lambda d: THEME_COLORS.get(d, "#666666"))
  ax.barh(df["Theme"], df["Effect"], color=colors)
  for i, (_, row) in enumerate(df.iterrows()):
    ax.text(row["Effect"], i, f"  NES≈{row['Effect']:.2f}\n  q={row['q_value']:.3g}", va="center", fontsize=9)
  ax.set_xlabel("Mean NES across representative pathways")
  ax.set_title("Figure 2. Condensed Themes Overview", fontsize=14, pad=16)
  fig.tight_layout()
  fig.savefig(pdf_path, dpi=300)
  fig.savefig(png_path, dpi=300)
  plt.close(fig)
  return pdf_path, png_path, "Figure 2. Condensed Themes Overview"


def plot_verdict_heatmap(output_dir: Path, claim_reviews: List[dict]) -> tuple[Path, Path, str]:
  fig_dir = ensure_dir(output_dir / "figures")
  base = fig_dir / "figure03_verdict_heatmap"
  pdf_path = base.with_suffix(".pdf")
  png_path = base.with_suffix(".png")
  if not claim_reviews:
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, "No claim reviews available.", ha="center", va="center")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path, "Figure 3. Verdict Matrix After Micro-Revision"
  verdict_order = ["Supported", "Partial", "Hypothesis", "Not supported"]
  rows = []
  for review in claim_reviews:
    label = f"{review['theme_id']} | {review.get('helper_name') or 'theme-only'}"
    verdict = review.get("verdict", "Hypothesis")
    row = {v: 0 for v in verdict_order}
    verdict_clean = verdict.title()
    if verdict_clean not in row:
      verdict_clean = "Hypothesis"
    row[verdict_clean] = 1
    row["Label"] = label
    rows.append(row)
  df = pd.DataFrame(rows).set_index("Label")

  fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.3)))
  sns.heatmap(df, cmap="BuPu", linewidths=0.5, linecolor="white", cbar=False, ax=ax)
  ax.set_title("Figure 3. Verdict Matrix After Micro-Revision")
  fig.tight_layout()
  fig.savefig(pdf_path, dpi=300)
  fig.savefig(png_path, dpi=300)
  plt.close(fig)
  return pdf_path, png_path, "Figure 3. Verdict Matrix After Micro-Revision"


PREDICTION_CATEGORIES = [
    ("Chromatin/Enhancer", ("chromatin", "h3k27", "atac", "enhancer", "chip")),
    ("Gene Expression", ("rna", "transcript", "expression", "bulk", "single-cell")),
    ("Functional Perturbation", ("crispr", "perturb", "knockdown", "overexpress", "inhibit", "block")),
    ("Phenotypic/Other", ()),
]


def categorize_prediction(pred: str) -> str:
  lower = pred.lower()
  for category, keywords in PREDICTION_CATEGORIES:
    if any(keyword in lower for keyword in keywords):
      return category
  return "Phenotypic/Other"


def plot_prediction_flow(output_dir: Path, evidence_csv: Path, themes: Dict[str, List[dict]]) -> tuple[Path, Path, str]:
  fig_dir = ensure_dir(output_dir / "figures")
  base = fig_dir / "figure04_prediction_flow"
  pdf_path = base.with_suffix(".pdf")
  png_path = base.with_suffix(".png")
  if not evidence_csv.exists():
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, "Evidence box not found.", ha="center", va="center")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path, "Figure 4. Follow-up Experiment Flow"
  df = pd.read_csv(evidence_csv)
  if df.empty:
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, "Evidence box is empty.", ha="center", va="center")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path, "Figure 4. Follow-up Experiment Flow"

  theme_order = []
  for direction in ("UP", "DOWN"):
    for theme in themes.get(direction, []):
      theme_order.append(theme["theme_id"])
  if not theme_order:
    theme_order = list(df["theme_id"])

  flow_counts: Dict[tuple[str, str], int] = {}
  predictions_by_theme: Dict[str, List[str]] = {}
  for _, row in df.iterrows():
    theme_id = row.get("theme_id", "theme")
    theme_label = row.get("theme_label", theme_id)
    predictions_by_theme[theme_id] = predictions_by_theme.get(theme_id, [])
    for idx in (1, 2):
      pred = row.get(f"prediction_{idx}")
      if isinstance(pred, str) and pred.strip():
        category = categorize_prediction(pred)
        flow_counts[(theme_id, category)] = flow_counts.get((theme_id, category), 0) + 1
        predictions_by_theme[theme_id].append(pred.strip())

  if not flow_counts:
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, "No predictions available for visualization.", ha="center", va="center")
    return path, fig

  unique_categories = [cat for cat, _ in PREDICTION_CATEGORIES]
  theme_positions = {theme_id: idx for idx, theme_id in enumerate(theme_order)}
  category_positions = {cat: idx for idx, cat in enumerate(unique_categories)}

  fig, ax = plt.subplots(figsize=(10, max(5, len(theme_order) * 0.4)))
  for theme_id, idx in theme_positions.items():
    label = next((theme["label"] for dir_values in themes.values() for theme in dir_values if theme["theme_id"] == theme_id), theme_id)
    ax.scatter(0, idx, color="#444444", s=80)
    ax.text(-0.02, idx, label, ha="right", va="center", fontsize=9)
  for category, idx in category_positions.items():
    ax.scatter(1, idx, color="#8888ff", s=90)
    ax.text(1.02, idx, category, ha="left", va="center", fontsize=10)

  for (theme_id, category), count in flow_counts.items():
    if theme_id not in theme_positions or category not in category_positions:
      continue
    y0 = theme_positions[theme_id]
    y1 = category_positions[category]
    linewidth = 0.8 + count * 0.9
    ax.plot([0, 1], [y0, y1], color="#bbbbdd", linewidth=linewidth, alpha=0.85)

  ax.set_xlim(-0.2, 1.2)
  ax.set_ylim(-1, max(len(theme_positions), len(category_positions)) + 0.5)
  ax.axis("off")
  ax.set_title("Figure 4. Follow-up Experiment Flow (theme → assay)", fontsize=14, pad=20)
  fig.tight_layout()
  fig.savefig(pdf_path, dpi=300)
  fig.savefig(png_path, dpi=300)
  plt.close(fig)
  return pdf_path, png_path, "Figure 4. Follow-up Experiment Flow"


def plot_consistency_scatter(output_dir: Path, themes: Dict[str, List[dict]]) -> tuple[Path, Path, str]:
  fig_dir = ensure_dir(output_dir / "figures")
  base = fig_dir / "figure05_gsea_vs_helper_nes"
  pdf_path = base.with_suffix(".pdf")
  png_path = base.with_suffix(".png")
  records = []
  for direction, theme_list in themes.items():
    for theme in theme_list:
      helper_mean = theme.get("helper_mean_effect")
      if helper_mean is None:
        continue
      records.append({
          "Theme": theme.get("label", theme.get("theme_id", "theme")),
          "Direction": direction,
          "GSEA": theme.get("effect", 0.0),
          "Helper": helper_mean,
      })
  if not records:
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, "No helper mean NES available.", ha="center", va="center")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path, "Figure 5. GSEA vs Helper NES (n/a)"
  df = pd.DataFrame(records)
  fig, ax = plt.subplots(figsize=(6.5, 6))
  colors = df["Direction"].map(lambda d: THEME_COLORS.get(d, "#555555"))
  ax.scatter(df["GSEA"], df["Helper"], c=colors, s=80, edgecolor="black")
  lim = max(max(df["GSEA"].abs().max(), df["Helper"].abs().max()), 0.1)
  ax.plot([-lim, lim], [-lim, lim], linestyle="--", color="#aaaaaa", linewidth=1)
  for _, row in df.iterrows():
    ax.text(row["GSEA"], row["Helper"], row["Theme"], fontsize=8, ha="left", va="bottom")
  ax.set_xlabel("Mean GSEA NES")
  ax.set_ylabel("Mean helper NES")
  ax.set_title("Figure 5. GSEA vs Helper NES consistency")
  fig.tight_layout()
  fig.savefig(pdf_path, dpi=300)
  fig.savefig(png_path, dpi=300)
  plt.close(fig)
  return pdf_path, png_path, "Figure 5. GSEA vs Helper NES consistency"


def main(argv: List[str] | None = None) -> None:
  parser = argparse.ArgumentParser(description="Generate figures and a PDF report from agent outputs.")
  parser.add_argument("--outputs-dir", required=True, type=Path, help="Directory containing agent outputs (JSON, CSV, thesis).")
  args = parser.parse_args(argv)

  output_dir = args.outputs_dir
  themes = load_json(output_dir / "themes_condensed.json")
  claims = load_json(output_dir / "helper_claims.json")
  claim_reviews = load_json(output_dir / "claim_reviews.json")
  helpers_meta = load_json(output_dir / "helpers.json")
  evidence_box = output_dir / "evidence_box.csv"

  png_entries = []
  _, png_path, caption = plot_theme_helper_links(output_dir, themes, claims, helpers_meta, claim_reviews)
  png_entries.append((caption, png_path))
  _, png_path, caption = plot_theme_summary(output_dir, themes, claim_reviews)
  png_entries.append((caption, png_path))
  _, png_path, caption = plot_verdict_heatmap(output_dir, claim_reviews)
  png_entries.append((caption, png_path))
  _, png_path, caption = plot_prediction_flow(output_dir, evidence_box, themes)
  png_entries.append((caption, png_path))
  _, png_path, caption = plot_consistency_scatter(output_dir, themes)
  png_entries.append((caption, png_path))

  thesis_path = output_dir / "mini_thesis.txt"
  build_markdown_pdf(output_dir, thesis_path, png_entries)


def build_markdown_pdf(output_dir: Path,
                       thesis_path: Path,
                       figure_entries: List[Tuple[str, Path]]) -> None:
  md_path = output_dir / "mini_thesis.md"
  thesis_text = thesis_path.read_text(encoding="utf-8") if thesis_path.exists() else "Mini-thesis not found."
  md_lines = ["# Mini-Thesis", "", thesis_text.strip(), "", "## Figures"]
  for caption, png_path in figure_entries:
    rel_path = png_path.relative_to(output_dir)
    md_lines.append(f"![{caption}]({rel_path.as_posix()})")
    md_lines.append("")
  md_path.write_text("\n".join(md_lines), encoding="utf-8")
  md_to_pdf(md_path, output_dir / "mini_thesis_report.pdf")


if __name__ == "__main__":
  main()
