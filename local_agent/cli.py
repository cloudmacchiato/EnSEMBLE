"""Command-line interface for the Gemini-backed agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .background import parse_background_txt
from .config import AnalysisSettings, GeminiConfig
from .data_models import ClaimEvidence, GSEARecord, HelperClaim, ThemeSummary
from .llm import GeminiLLM
from .pipeline import Pipeline, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Gemini-powered ESEA↔GSEA interpretability agent.")
    parser.add_argument("--gsea-csv", required=True, type=Path, help="Path to the GSEA results CSV.")
    parser.add_argument(
        "--esea-csv",
        type=Path,
        default=None,
        help="Path to the enhancer helper CSV (esea_helpers.csv). Omit for GSEA-only runs.",
    )
    parser.add_argument("--background-txt", required=True, type=Path, help="Path to the filled background text form.")
    parser.add_argument(
        "--output-dir",
        default=Path("outputs/gemini_agent"),
        type=Path,
        help="Directory for generated artefacts.",
    )
    parser.add_argument(
        "--gemini-model",
        default="models/gemini-2.5-flash",
        help="Gemini model identifier to call (see ai.google.dev for model catalogue).",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Gemini API key (omit when GOOGLE_API_KEY is already exported).",
    )
    parser.add_argument(
        "--critic-gemini-model",
        default=None,
        help="Gemini model identifier for the critic head (defaults to --gemini-model).",
    )
    parser.add_argument("--disable-critic", action="store_true", help="Disable the critic review stage.")
    parser.add_argument("--gsea-only", action="store_true", help="Run without enhancer (ESEA) data; outputs will be marked as hypotheses.")
    parser.add_argument("--gemini-temperature", default=0.3, type=float, help="Sampling temperature for Gemini.")
    parser.add_argument("--gemini-top-p", default=0.95, type=float, help="Top-p nucleus sampling for Gemini.")
    parser.add_argument("--gemini-top-k", default=32, type=int, help="Top-k sampling for Gemini.")
    parser.add_argument(
        "--gemini-max-output-tokens",
        default=10000,
        type=int,
        help="Maximum number of tokens Gemini may generate per call.",
    )
    parser.add_argument("--gsea-top-n", default=0, type=int, help="Maximum pathways per direction to send to Gemini (0 keeps all).")
    parser.add_argument("--gsea-q-threshold", default=0.05, type=float, help="Q-value cutoff for GSEA pathways.")
    parser.add_argument("--esea-q-threshold", default=0.05, type=float, help="Helper q-value cutoff used for ranking.")
    parser.add_argument(
        "--esea-effect-threshold",
        default=0.25,
        type=float,
        help="Helper NES threshold used when ranking helper confidence.",
    )
    parser.add_argument("--theme-cap", default=10, type=int, help="Maximum number of themed pathway groups per direction.")
    parser.add_argument(
        "--theme-top-pathways",
        default=3,
        type=int,
        help="Representative pathways per theme to surface to Gemini.",
    )
    parser.add_argument(
        "--theme-leading-edge-target",
        default=12,
        type=int,
        help="Target number of leading-edge genes per theme (10–15 recommended).",
    )
    parser.add_argument(
        "--theme-cap-total",
        default=20,
        type=int,
        help="Maximum themes across both directions (set high to keep all helper-supported themes).",
    )
    parser.add_argument(
        "--helper-claims-per-theme",
        default=0,
        type=int,
        help="Maximum helper links per theme supplied to the LLM (0 disables the cap).",
    )
    parser.add_argument(
        "--esea-max-per-direction",
        default=50,
        type=int,
        help="Maximum helper rows per direction to send to Gemini.",
    )
    parser.add_argument(
        "--resume-stage",
        choices=["mini_thesis"],
        help="Resume a single failed stage using cached outputs (currently supports mini_thesis).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.resume_stage:
        if args.esea_csv is None:
            if not args.gsea_only:
                parser.error("Missing --esea-csv. Provide the helper CSV or pass --gsea-only for pathway-only interpretation.")
        else:
            if args.gsea_only:
                parser.error("--gsea-only cannot be combined with --esea-csv.")

    gemini_config = GeminiConfig(
        model=args.gemini_model,
        api_key=args.gemini_api_key,
        temperature=args.gemini_temperature,
        top_p=args.gemini_top_p,
        top_k=args.gemini_top_k,
        max_output_tokens=args.gemini_max_output_tokens,
    )

    if args.resume_stage:
        llm = GeminiLLM(gemini_config)
        _resume_stage(args, llm)
        return

    analysis_settings = AnalysisSettings(
        gsea_top_n=args.gsea_top_n,
        gsea_q_threshold=args.gsea_q_threshold,
        helper_q_threshold=args.esea_q_threshold,
        helper_nes_threshold=args.esea_effect_threshold,
        helper_max_per_direction=args.esea_max_per_direction,
        helper_claims_per_theme=args.helper_claims_per_theme,
        theme_cap=args.theme_cap,
        theme_top_pathways=args.theme_top_pathways,
        theme_leading_edge_target=args.theme_leading_edge_target,
        theme_cap_total=args.theme_cap_total,
    )

    llm = GeminiLLM(gemini_config)
    critic_llm: GeminiLLM | None = None
    if not args.disable_critic:
        critic_config = GeminiConfig(
            model=args.critic_gemini_model or args.gemini_model,
            api_key=args.gemini_api_key,
            temperature=args.gemini_temperature,
            top_p=args.gemini_top_p,
            top_k=args.gemini_top_k,
            # Allow critic head to emit the full requested budget (no 4k clamp).
            max_output_tokens=args.gemini_max_output_tokens,
        )
        critic_llm = GeminiLLM(critic_config)

    artefacts = run_pipeline(
        gsea_csv=args.gsea_csv,
        esea_csv=args.esea_csv,
        background_txt=args.background_txt,
        output_dir=args.output_dir,
        llm=llm,
        critic_llm=critic_llm,
        analysis_settings=analysis_settings,
        gsea_only=args.gsea_only,
    )

    print(f"Mini-thesis written to {args.output_dir / 'mini_thesis.txt'}")
    print(f"Helper claims saved to {args.output_dir / 'helper_claims.json'}")
    print(f"Evidence box saved to {args.output_dir / 'evidence_box.csv'}")
    print(f"Revision notes saved to {args.output_dir / 'revision_notes.json'}")
    print(f"Total claims: {len(artefacts.helper_claims)} | Helper evidence available: {artefacts.helpers_available}")


def _resume_stage(args, llm: GeminiLLM) -> None:
    if args.resume_stage == "mini_thesis":
        _resume_mini_thesis(args.output_dir, args.background_txt, llm)
    else:
        raise ValueError(f"Unsupported resume stage: {args.resume_stage}")


def _resume_mini_thesis(output_dir: Path, background_txt: Path, llm: GeminiLLM) -> None:
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    background = parse_background_txt(background_txt)
    themes = _load_themes(output_dir / "themes_condensed.json")
    helper_claims = _load_helper_claims(output_dir / "helper_claims.json")
    claim_evidence = _load_claim_evidence(output_dir / "claim_reviews.json", helper_claims)
    helpers_available = _load_helpers_status(output_dir / "helpers_status.txt")
    pipeline = Pipeline(llm=llm)
    mini_thesis = pipeline._compose_mini_thesis(
        themes=themes,
        claim_evidence=claim_evidence,
        background=background,
        helpers_available=helpers_available,
    )
    (output_dir / "mini_thesis.txt").write_text(mini_thesis, encoding="utf-8")
    _append_stage_failures(output_dir / "stage_failures.json", pipeline.stage_failures)
    print(f"Mini-thesis regenerated at {output_dir / 'mini_thesis.txt'}")


def _load_themes(path: Path) -> dict[str, list[ThemeSummary]]:
    if not path.exists():
        raise FileNotFoundError(f"Theme summary not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    themes: dict[str, list[ThemeSummary]] = {}
    for direction, entries in raw.items():
        theme_list: list[ThemeSummary] = []
        for entry in entries:
            top_records: list[GSEARecord] = []
            for rec in entry.get("top_pathways", []):
                top_records.append(
                    GSEARecord(
                        term=rec.get("term", ""),
                        source=rec.get("collection"),
                        nes=rec.get("nes", 0.0),
                        q_value=rec.get("q_value", 1.0),
                        size=rec.get("size", 0),
                        direction=direction,
                        score=rec.get("nes", 0.0),
                        leading_edge=tuple(),
                    )
                )
            theme_list.append(
                ThemeSummary(
                    theme_id=entry.get("theme_id", ""),
                    label=entry.get("label", ""),
                    direction=direction,
                    collection=entry.get("collection"),
                    effect=entry.get("effect", 0.0),
                    q_value=entry.get("q_value", 1.0),
                    top_pathways=top_records,
                    leading_edges=tuple(entry.get("leading_edges", ())),
                    helper_mean_effect=entry.get("helper_mean_effect"),
                )
            )
        themes[direction] = theme_list
    return themes


def _load_helper_claims(path: Path) -> list[HelperClaim]:
    if not path.exists():
        raise FileNotFoundError(f"Helper claims not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    claims: list[HelperClaim] = []
    for entry in raw:
        claims.append(
            HelperClaim(
                theme_id=entry.get("theme_id", ""),
                theme_label=entry.get("theme_label", ""),
                helper_name=entry.get("helper_name"),
                helper_class=entry.get("helper_class", "theme_only"),
                direction=entry.get("direction", "").upper() or "UP",
                function_phrases=tuple(entry.get("function_phrases") or ()),
                rationale=entry.get("rationale", ""),
                confidence=entry.get("confidence", "medium"),
                helper_top_hallmark=entry.get("helper_top_hallmark"),
                helper_effect=entry.get("helper_effect"),
                helper_q_value=entry.get("helper_q_value"),
                claim_id=entry.get("claim_id"),
            )
        )
    return claims


def _load_claim_evidence(path: Path, helper_claims: list[HelperClaim]) -> list[ClaimEvidence]:
    if not path.exists():
        raise FileNotFoundError(f"Claim reviews not found: {path}")
    claim_lookup = {(claim.theme_id, claim.helper_name): claim for claim in helper_claims}
    raw = json.loads(path.read_text(encoding="utf-8"))
    results: list[ClaimEvidence] = []
    for entry in raw:
        key = (entry.get("theme_id"), entry.get("helper_name"))
        claim = claim_lookup.get(key)
        if claim is None:
            continue
        predictions_raw = entry.get("predictions") or []
        if isinstance(predictions_raw, str):
            predictions = (predictions_raw.strip(),)
        else:
            predictions = tuple(str(item).strip() for item in predictions_raw if str(item).strip())
        evidence_snippets = entry.get("evidence_snippets") or []
        results.append(
            ClaimEvidence(
                claim=claim,
                evidence_snippets=tuple(evidence_snippets),
                alternative=entry.get("alternative", ""),
                gaps=entry.get("gaps", ""),
                predictions=predictions,
                verdict=entry.get("verdict", "Hypothesis"),
                verdict_reason=entry.get("verdict_reason", ""),
            )
        )
    return results


def _load_helpers_status(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8").lower()
    return "available" in text


def _append_stage_failures(path: Path, new_entries: list[dict]) -> None:
    if not new_entries:
        return
    existing: list[dict] = []
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
    existing.extend(new_entries)
    path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
