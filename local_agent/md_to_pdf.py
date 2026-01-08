"""Render Markdown into PDF with a simple CSS theme."""

from __future__ import annotations

import sys
from pathlib import Path

from markdown import markdown
from weasyprint import HTML, CSS

CSS_STRING = """
@page { size: Letter; margin: 1in; }
body { font-family: "DejaVu Sans", Arial, sans-serif; line-height: 1.5; font-size: 12pt; }
h1, h2, h3 { margin-top: 1.2em; font-weight: bold; }
code, pre { font-family: "DejaVu Sans Mono", monospace; font-size: 0.9em; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 0.35em 0.6em; }
img { max-width: 100%; margin: 0.8em 0; }
"""


def md_to_pdf(input_md: Path, output_pdf: Path) -> None:
    """Convert a Markdown file to PDF with WeasyPrint."""
    md_text = input_md.read_text(encoding="utf-8")
    html = markdown(
        md_text,
        extensions=[
            "extra",
            "toc",
            "fenced_code",
            "codehilite",
        ],
    )
    HTML(string=html, base_url=str(input_md.parent)).write_pdf(
        str(output_pdf),
        stylesheets=[CSS(string=CSS_STRING)],
    )
    print(f"Wrote {output_pdf}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m local_agent.md_to_pdf input.md output.pdf")
        raise SystemExit(1)
    md_to_pdf(Path(sys.argv[1]), Path(sys.argv[2]))
