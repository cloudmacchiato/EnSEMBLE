"""End-to-end orchestration for the local agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .background import Background, parse_background_txt
from .data_models import Theme
from .llm import GeminiLLM, ensure_json
from .prefilter import (
    load_esea,
    load_gsea,
    split_gsea_by_direction,
    to_esea_records,
    to_gsea_records,
)
from .prompts import (
    BACKGROUND_CHECK_PROMPT,
    CLAIM_EXTRACTION_PROMPT,
    DRAFT_CONNECTIONS_PROMPT,
    FINAL_SUMMARY_PROMPT,
    GLOBAL_SYSTEM_PROMPT,
    REVISION_PROMPT,
)
from .themes import group_by_theme
from .verification import (
    Claim,
    VerificationResult,
    apply_background_flag,
    decide_support,
    index_esea,
    index_themes,
    normalize,
)


@dataclass
class PipelineArtifacts:
    background: Background
    themes: Dict[str, List[Theme]]
    draft_connections: List[dict]
    claims: List[Claim]
    verification: List[VerificationResult]
    revised_connections: List[dict]
    mini_thesis: str


class Pipeline:
    def __init__(self, llm: GeminiLLM):
        self.llm = llm

    def run(
        self,
        gsea_csv: Path,
        esea_csv: Path,
        background_txt: Path,
        output_dir: Path,
    ) -> PipelineArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)

        background = parse_background_txt(background_txt)

        gsea_rows = load_gsea(gsea_csv)
        esea_rows = load_esea(esea_csv)

        direction_tables = split_gsea_by_direction(gsea_rows)
        themes = {
            direction: group_by_theme(to_gsea_records(rows, direction), direction)
            for direction, rows in direction_tables.items()
        }

        draft_connections = self._draft_connections(themes, esea_rows, background)
        claims = self._extract_claims(draft_connections)
        verification = self._verify(claims, themes, esea_rows, background)
        revised_connections = self._revise_connections(verification)
        mini_thesis = self._summarize(revised_connections, background)

        self._write_artifacts(output_dir, draft_connections, claims, verification, revised_connections, mini_thesis)

        return PipelineArtifacts(
            background=background,
            themes=themes,
            draft_connections=draft_connections,
            claims=claims,
            verification=verification,
            revised_connections=revised_connections,
            mini_thesis=mini_thesis,
        )

    def _draft_connections(self, themes: Dict[str, List[Theme]], esea_rows: List[dict], background: Background) -> List[dict]:
        payload = {
            "themes": {
                direction: [
                    {
                        "label": theme.label,
                        "direction": theme.direction,
                        "top_terms": [rec.term for rec in theme.top_terms(5)],
                    }
                    for theme in theme_list
                ]
                for direction, theme_list in themes.items()
            },
            "esea": esea_rows,
            "background": background.summary,
        }
        response = self.llm(GLOBAL_SYSTEM_PROMPT, f"{DRAFT_CONNECTIONS_PROMPT}\nInput JSON:\n{json.dumps(payload)}")
        return ensure_json(response)

    def _extract_claims(self, draft_connections: List[dict]) -> List[Claim]:
        response = self.llm(
            GLOBAL_SYSTEM_PROMPT,
            f"{CLAIM_EXTRACTION_PROMPT}\nInput JSON:\n{json.dumps(draft_connections)}",
        )
        raw_items = ensure_json(response)
        return [Claim(**item) for item in raw_items]

    def _verify(
        self,
        claims: List[Claim],
        themes: Dict[str, List[Theme]],
        esea_rows: List[dict],
        background: Background,
    ) -> List[VerificationResult]:
        esea_records = to_esea_records(esea_rows)
        esea_index = index_esea(esea_records)
        theme_indices = {
            direction: index_themes(theme_list)
            for direction, theme_list in themes.items()
        }

        results: List[VerificationResult] = []
        for claim in claims:
            direction = claim.direction.upper()
            theme_index = theme_indices.get(direction, {})
            matched_theme = theme_index.get(normalize(claim.theme))
            matched_terms = [rec.term for rec in matched_theme.top_terms(5)] if matched_theme else []

            if not matched_theme:
                decision = "Refuted"
                reason = "Theme not present among top-ranked pathways for this direction."
            else:
                matched_esea = esea_index.get(normalize(claim.cell_type))
                decision, reason = decide_support(matched_esea, claim.direction)

            matched_esea = esea_index.get(normalize(claim.cell_type))

            bg_prompt = BACKGROUND_CHECK_PROMPT.format(background=background.summary, claim=claim.claim)
            background_flag = self.llm(GLOBAL_SYSTEM_PROMPT, bg_prompt).content.strip()
            decision, reason, background_flag = apply_background_flag(decision, reason, background_flag)

            results.append(
                VerificationResult(
                    claim=claim,
                    decision=decision,
                    reason=reason,
                    matched_esea=matched_esea,
                    matched_terms=matched_terms,
                    background_flag=background_flag,
                )
            )
        return results

    def _revise_connections(self, verification: List[VerificationResult]) -> List[dict]:
        table = [result.to_row() for result in verification]
        response = self.llm(
            GLOBAL_SYSTEM_PROMPT,
            f"{REVISION_PROMPT}\nVerification Table:\n{json.dumps(table)}",
        )
        return ensure_json(response)

    def _summarize(self, revised_connections: List[dict], background: Background) -> str:
        response = self.llm(
            GLOBAL_SYSTEM_PROMPT,
            f"{FINAL_SUMMARY_PROMPT}\nConnections JSON:\n{json.dumps(revised_connections)}\nBackground:\n{background.summary}",
        )
        return response.content.strip()

    def _write_artifacts(
        self,
        output_dir: Path,
        draft_connections: List[dict],
        claims: List[Claim],
        verification: List[VerificationResult],
        revised_connections: List[dict],
        mini_thesis: str,
    ) -> None:
        (output_dir / "draft_connections.json").write_text(json.dumps(draft_connections, indent=2), encoding="utf-8")
        claims_payload = [claim.__dict__ for claim in claims]
        (output_dir / "claims.json").write_text(json.dumps(claims_payload, indent=2), encoding="utf-8")
        verification_payload = [
            {
                **result.to_row(),
                "background_flag": result.background_flag,
                "matched_terms": result.matched_terms,
            }
            for result in verification
        ]
        (output_dir / "verification.json").write_text(json.dumps(verification_payload, indent=2), encoding="utf-8")
        (output_dir / "revised_connections.json").write_text(json.dumps(revised_connections, indent=2), encoding="utf-8")
        (output_dir / "mini_thesis.txt").write_text(mini_thesis, encoding="utf-8")

        if verification:
            headers = ["theme", "direction", "cell_type", "decision", "effect_size", "q_value", "reason"]
            rows = [result.to_row() for result in verification]
            widths = {header: len(header) for header in headers}
            for row in rows:
                for header in headers:
                    widths[header] = max(widths[header], len(str(row[header])))
            lines = []
            header_line = " | ".join(header.ljust(widths[header]) for header in headers)
            separator = "-+-".join("-" * widths[header] for header in headers)
            lines.append(header_line)
            lines.append(separator)
            for row in rows:
                lines.append(" | ".join(str(row[header]).ljust(widths[header]) for header in headers))
            (output_dir / "verification_table.txt").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(
    gsea_csv: Path,
    esea_csv: Path,
    background_txt: Path,
    output_dir: Path,
    llm: GeminiLLM,
) -> PipelineArtifacts:
    pipeline = Pipeline(llm=llm)
    return pipeline.run(gsea_csv, esea_csv, background_txt, output_dir)
