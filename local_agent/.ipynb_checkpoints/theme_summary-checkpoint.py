"""CLI to emit full theme summaries (without caps) from a GSEA CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from .config import AnalysisSettings
from .prefilter import load_gsea, split_gsea_by_direction, to_gsea_records
from .themes import build_theme_summaries, ThemeSummary


def _theme_to_dict(theme: ThemeSummary) -> Dict[str, object]:
    return {
        "theme_id": theme.theme_id,
        "label": theme.label,
        "direction": theme.direction,
        "collection": theme.collection,
        "effect": theme.effect,
        "q_value": theme.q_value,
        "leading_edges": list(theme.leading_edges),
        "top_pathways": [
            {
                "term": record.term,
                "collection": record.source,
                "nes": record.nes,
                "q_value": record.q_value,
                "size": record.size,
            }
            for record in theme.top_pathways
        ],
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gsea-csv", required=True, type=Path, help="Path to GSEA results CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("theme_outputs"),
        help="Directory to write theme JSON (default: ./theme_outputs)",
    )
    parser.add_argument(
        "--gsea-q-threshold",
        type=float,
        default=0.05,
        help="Q-value cutoff applied before theme condensation.",
    )
    parser.add_argument(
        "--theme-cap-per-direction",
        type=int,
        default=1000,
        help="Maximum themes per direction (set high to disable).",
    )
    parser.add_argument(
        "--theme-cap-total",
        type=int,
        default=2000,
        help="Maximum total themes to keep (set high to effectively disable).",
    )
    args = parser.parse_args(argv)

    rows = load_gsea(args.gsea_csv)
    settings = AnalysisSettings(
        gsea_q_threshold=args.gsea_q_threshold,
        theme_cap=max(args.theme_cap_per_direction, 1),
        theme_cap_total=max(args.theme_cap_total, 1),
    )
    buckets = split_gsea_by_direction(rows, settings)
    records = {direction: to_gsea_records(vals, direction) for direction, vals in buckets.items()}
    themes = build_theme_summaries(records, settings)

    payload = {
        direction: [_theme_to_dict(theme) for theme in theme_list]
        for direction, theme_list in themes.items()
    }

    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "themes.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
