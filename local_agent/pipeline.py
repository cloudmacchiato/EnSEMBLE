"""End-to-end orchestration for the local agent."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .config import AnalysisSettings

from .background import Background, parse_background_txt
from .data_models import ClaimEvidence, HelperClaim, HelperRecord, ThemeSummary, EvidenceBundle
from .llm import GeminiLLM, LLMError, ensure_json
from .prefilter import (
    load_gsea,
    load_helpers,
    get_default_tf_helper_names,
    split_gsea_by_direction,
    to_gsea_records,
)
from .prompts import (
    EVIDENCE_RETRIEVAL_PROMPT,
    CRITIC_PROMPT,
    GLOBAL_SYSTEM_PROMPT,
    HELPER_LINKER_PROMPT,
    MINI_THESIS_PROMPT,
)
from .themes import build_theme_summaries


BACKGROUND_STOPWORDS = {
    "cell",
    "cells",
    "program",
    "programs",
    "signal",
    "signals",
    "pathway",
    "pathways",
    "response",
    "responses",
    "additional",
    "theme",
    "regulation",
    "regulatory",
}


def _slugify_text(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", label).strip("-")
    return slug.lower() or "theme"


@dataclass
class PipelineArtifacts:
    background: Background
    themes: Dict[str, List[ThemeSummary]]
    helpers: List[HelperRecord]
    helpers_available: bool
    helper_claims: List[HelperClaim]
    claim_evidence: List[ClaimEvidence]
    mini_thesis: str
    revision_notes: List[dict]
    stage_failures: List[dict]


class Pipeline:
    def __init__(
        self,
        llm: GeminiLLM,
        analysis_settings: AnalysisSettings | None = None,
        critic_llm: GeminiLLM | None = None,
    ):
        self.llm = llm
        self.analysis = analysis_settings or AnalysisSettings()
        self.critic_llm = critic_llm
        self.tf_helper_names = get_default_tf_helper_names()
        self.stage_failures: List[dict] = []

    def run(
        self,
        gsea_csv: Path,
        esea_csv: Path | None,
        background_txt: Path,
        output_dir: Path,
        gsea_only: bool = False,
    ) -> PipelineArtifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        self.stage_failures = []

        background = parse_background_txt(background_txt)
        gsea_rows = load_gsea(gsea_csv)
        direction_tables = split_gsea_by_direction(gsea_rows, self.analysis)
        gsea_records = {
            direction: to_gsea_records(rows, direction) for direction, rows in direction_tables.items()
        }

        helpers_available = not gsea_only and esea_csv is not None
        helper_records: List[HelperRecord] = []
        if helpers_available and esea_csv is not None:
            helper_records = load_helpers(esea_csv, self.analysis, self.tf_helper_names)
            helpers_available = bool(helper_records)

        head_linker_settings = replace(
            self.analysis,
            theme_cap=self.analysis.head_linker_theme_cap,
            theme_cap_total=self.analysis.head_linker_theme_cap_total,
        )
        theme_candidates = build_theme_summaries(gsea_records, head_linker_settings)

        helper_claims = self._generate_helper_claims(
            theme_candidates,
            helper_records,
            background,
            helpers_available,
        )
        helper_claims = self._dedupe_claims(helper_claims)
        helper_claims = self._retarget_helper_claims(helper_claims, theme_candidates)
        helper_claims = self._ensure_background_theme_claims(
            helper_claims, theme_candidates, background
        )
        helper_claims = self._ensure_direction_coverage(helper_claims, theme_candidates)
        helper_claims = self._prioritize_helper_claims(helper_claims)
        background_tokens = self._collect_background_tokens(background)

        helper_claims, themes = self._select_themes_from_candidates(
            theme_candidates, helper_claims, background_tokens, helper_records
        )
        evidence_bundles, evidence_head_raw = self._gather_evidence(
            helper_claims,
            themes,
            helper_records,
            background,
            helpers_available,
        )
        claim_evidence = self._critique_claims(
            helper_claims,
            evidence_bundles,
            background,
            helpers_available,
        )
        helper_claims, claim_evidence, revision_notes = self._revise_claims(
            helper_claims,
            claim_evidence,
            themes,
            helper_records,
            background,
            helpers_available,
        )
        helper_claims, claim_evidence = self._dedupe_claims_with_evidence(
            helper_claims,
            claim_evidence,
        )
        self._update_theme_helper_effects(themes, helper_claims)
        mini_thesis = self._compose_mini_thesis(
            themes,
            claim_evidence,
            background,
            helpers_available,
        )

        self._write_artifacts(
            output_dir=output_dir,
            themes=themes,
            helpers=helper_records,
            claims=helper_claims,
            claim_evidence=claim_evidence,
            evidence_head_raw=evidence_head_raw,
            mini_thesis=mini_thesis,
            helpers_available=helpers_available,
            revision_notes=revision_notes,
            stage_failures=list(self.stage_failures),
        )

        return PipelineArtifacts(
            background=background,
            themes=themes,
            helpers=helper_records,
            helpers_available=helpers_available,
            helper_claims=helper_claims,
            claim_evidence=claim_evidence,
            mini_thesis=mini_thesis,
            revision_notes=revision_notes,
            stage_failures=list(self.stage_failures),
        )

    # ------------------------------------------------------------------
    # Helper-linking head
    # ------------------------------------------------------------------

    STAGE_RETRY_ATTEMPTS = 2
    CRITIC_CHUNK_SIZE = 3
    # Allow all themes through to the mini-thesis (no per-direction cap).
    MINI_THESIS_THEME_CAP_PER_DIRECTION = 0
    MINI_THESIS_HELPER_THEME_PRIORITY = 3
    MINI_THESIS_CLAIM_CAP = 6
    LINKER_THEMES_UP_PER_BATCH = 2
    LINKER_THEMES_DOWN_PER_BATCH = 2
    LINKER_BATCH_HELPER_CAP = 3
    HELPER_CLAIM_LIMIT_PER_DIRECTION = 0
    THEME_ONLY_CAP_WITH_HELPERS = 1
    THEME_ONLY_CAP_NO_HELPERS = 3
    GLOBAL_CLAIM_CAP = 0
    EVIDENCE_BATCH_SIZE = 3

    def _call_llm_with_retry(
        self,
        stage_name: str,
        call_fn: Callable[[], Any],
        *,
        details: Optional[dict] = None,
        max_attempts: Optional[int] = None,
    ) -> tuple[Any | None, int, Optional[str]]:
        attempts = max_attempts or self.STAGE_RETRY_ATTEMPTS
        failures = 0
        last_error: Optional[str] = None
        for attempt in range(1, attempts + 1):
            try:
                result = call_fn()
                if failures:
                    self._record_stage_recovery(stage_name, failures, details=details)
                return result, failures, last_error
            except LLMError as exc:
                failures += 1
                last_error = str(exc)
                will_retry = attempt < attempts
                self._record_stage_failure(
                    stage_name=stage_name,
                    attempt=attempt,
                    error=last_error,
                    details=details,
                    will_retry=will_retry,
                )
                if not will_retry:
                    break
        return None, failures, last_error

    def _record_stage_failure(
        self,
        stage_name: str,
        attempt: int,
        error: str,
        *,
        details: Optional[dict],
        will_retry: bool,
    ) -> None:
        entry = {
            "stage": stage_name,
            "attempt": attempt,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "error": error,
            "will_retry": will_retry,
        }
        if details:
            entry["details"] = details
        self.stage_failures.append(entry)
        level = "WARN" if will_retry else "ERROR"
        action = "retrying" if will_retry else "no more retries; using fallback"
        print(f"[{level}] {stage_name}: attempt {attempt} failed ({error}); {action}.")

    def _record_stage_recovery(
        self,
        stage_name: str,
        failed_attempts: int,
        *,
        details: Optional[dict],
    ) -> None:
        entry = {
            "stage": stage_name,
            "status": "recovered",
            "failed_attempts": failed_attempts,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        if details:
            entry["details"] = details
        self.stage_failures.append(entry)
        print(f"[INFO] {stage_name}: recovered after {failed_attempts} failed attempt(s).")

    def _generate_helper_claims(
        self,
        themes: Dict[str, List[ThemeSummary]],
        helpers: List[HelperRecord],
        background: Background,
        helpers_available: bool,
    ) -> List[HelperClaim]:
        theme_batches = self._build_theme_batches(themes)
        background_tokens = self._collect_background_tokens(background)
        helper_payload = [
            {
                "helper_name": helper.helper_name,
                "helper_class": helper.helper_class,
                "tf_family": helper.tf_family,
                "direction": helper.direction,
                "nes": round(helper.nes, 3),
                "q_value": round(helper.q_value, 4),
                "size": helper.size,
                "top_hallmark": helper.top_hallmark,
            }
            for helper in helpers
        ]
        if not theme_batches:
            theme_batches = [[]]
        aggregated_claims: List[HelperClaim] = []
        for batch_index, theme_batch in enumerate(theme_batches, start=1):
            theme_payload = [self._theme_to_dict(theme) for theme in theme_batch]
            payload = {
                "background_summary": background.summary,
                "background_fields": background.as_dict(),
                "themes": theme_payload,
                "helpers": helper_payload,
                "helpers_available": helpers_available,
                "max_claims_per_batch": self.LINKER_BATCH_HELPER_CAP,
            }
            details = {
                "theme_batch_size": len(theme_payload),
                "helper_count": len(helper_payload),
                "batch_index": batch_index,
                "batch_total": len(theme_batches),
            }

            def _call_linker() -> Any:
                response = self.llm(
                    GLOBAL_SYSTEM_PROMPT,
                    f"{HELPER_LINKER_PROMPT}\nInput JSON:\n{json.dumps(payload)}",
                )
                return ensure_json(response)

            parsed, _, _ = self._call_llm_with_retry(
                "helper_linker",
                _call_linker,
                details=details,
            )
            if parsed is None:
                batch_claims: List[HelperClaim] = []
            else:
                batch_claims = self._parse_claims(
                    parsed,
                    themes,
                    helpers,
                    max_claims=self.LINKER_BATCH_HELPER_CAP if self.LINKER_BATCH_HELPER_CAP > 0 else None,
                )
                self._assign_missing_theme_ids(batch_claims, theme_batch)
            selected_helpers = self._select_batch_helper_claims(batch_claims)
            if selected_helpers:
                aggregated_claims.extend(selected_helpers)
            else:
                fallback_theme = self._select_background_theme(theme_batch, background_tokens)
                if fallback_theme is not None:
                    aggregated_claims.append(self._make_background_theme_claim(fallback_theme))
        if aggregated_claims:
            return aggregated_claims
        claims = self._fallback_claims(themes, helpers, helpers_available)
        return claims

    def _parse_claims(
        self,
        payload: List[dict],
        themes: Dict[str, List[ThemeSummary]],
        helpers: List[HelperRecord],
        max_claims: int | None = None,
    ) -> List[HelperClaim]:
        theme_index = {
            theme.theme_id: theme
            for direction in themes.values()
            for theme in direction
        }
        helper_index = {
            (helper.helper_name, helper.direction): helper for helper in helpers
        }
        claims: List[HelperClaim] = []
        if not isinstance(payload, list):
            return claims
        for item in payload:
            if not isinstance(item, dict):
                continue
            theme_id = item.get("theme_id")
            helper_name = item.get("helper_name")
            direction = item.get("direction", "").upper()
            theme = theme_index.get(theme_id)
            if not theme or (direction and direction != theme.direction):
                continue
            if helper_name:
                helper_ref = helper_index.get((helper_name, theme.direction))
                if helper_ref is None:
                    helper_ref = helper_index.get((helper_name, "UP")) or helper_index.get((helper_name, "DOWN"))
            else:
                helper_ref = None
            helper_class = item.get("helper_class") or (
                helper_ref.helper_class if helper_ref else "theme_only"
            )
            function_phrases = item.get("function_phrases") or []
            if isinstance(function_phrases, str):
                function_phrases = [function_phrases]
            confidence = (item.get("confidence") or "medium").lower()
            rationale = item.get("rationale") or ""
            claims.append(
                HelperClaim(
                    theme_id=theme.theme_id,
                    theme_label=theme.label,
                    helper_name=helper_name,
                    helper_class=helper_class,
                    direction=theme.direction,
                    function_phrases=tuple(str(p).strip() for p in function_phrases if str(p).strip()),
                    rationale=rationale,
                    confidence=confidence,
                    helper_top_hallmark=(
                        helper_ref.top_hallmark if helper_ref else item.get("helper_top_hallmark")
                    ),
                    helper_effect=helper_ref.nes if helper_ref else None,
                    helper_q_value=helper_ref.q_value if helper_ref else None,
                )
            )
            if max_claims and max_claims > 0 and len(claims) >= max_claims:
                break
        return claims

    def _fallback_claims(
        self,
        themes: Dict[str, List[ThemeSummary]],
        helpers: List[HelperRecord],
        helpers_available: bool,
    ) -> List[HelperClaim]:
        by_direction: Dict[str, List[HelperRecord]] = {"UP": [], "DOWN": []}
        for helper in helpers:
            by_direction.setdefault(helper.direction, []).append(helper)
        for bucket in by_direction.values():
            bucket.sort(key=lambda rec: (rec.q_value, -abs(rec.nes)))
        claims: List[HelperClaim] = []
        for direction in ("UP", "DOWN"):
            for theme in themes.get(direction, []):
                candidates = by_direction.get(direction, [])
                selected = candidates[: self.analysis.helper_claims_per_theme] if candidates else [None]
                for helper in selected:
                    if helper is None:
                        helper_name = None
                        helper_class = "theme_only"
                        hallmark = None
                        phrase = f"{theme.collection or 'pathway'} adaptation"
                        confidence = "low"
                    else:
                        helper_name = helper.helper_name
                        helper_class = helper.helper_class
                        hallmark = helper.top_hallmark
                        phrase = hallmark or f"{helper.helper_class} program"
                        confidence = "medium"
                    claims.append(
                        HelperClaim(
                            theme_id=theme.theme_id,
                            theme_label=theme.label,
                            helper_name=helper_name,
                            helper_class=helper_class,
                            direction=direction,
                            function_phrases=(phrase,),
                            rationale=(
                                f"{theme.label} ({direction}) aligns with {phrase}; "
                                f"helper q-values {'available' if helper else 'pending enhancer screen'}."
                            ),
                            confidence=confidence if helpers_available else "hypothesis",
                            helper_top_hallmark=hallmark,
                            helper_effect=helper.nes if helper else None,
                            helper_q_value=helper.q_value if helper else None,
                        )
                    )
        return claims

    def _build_theme_batches(self, themes: Dict[str, List[ThemeSummary]]) -> List[List[ThemeSummary]]:
        up_themes = list(themes.get("UP", []))
        down_themes = list(themes.get("DOWN", []))
        up_index = 0
        down_index = 0
        batches: List[List[ThemeSummary]] = []
        while up_index < len(up_themes) or down_index < len(down_themes):
            batch: List[ThemeSummary] = []
            if up_index < len(up_themes):
                batch.extend(
                    up_themes[up_index : up_index + self.LINKER_THEMES_UP_PER_BATCH]
                )
                up_index += self.LINKER_THEMES_UP_PER_BATCH
            if down_index < len(down_themes):
                batch.extend(
                    down_themes[down_index : down_index + self.LINKER_THEMES_DOWN_PER_BATCH]
                )
                down_index += self.LINKER_THEMES_DOWN_PER_BATCH
            batches.append(batch)
        return batches

    def _select_batch_helper_claims(self, claims: List[HelperClaim]) -> List[HelperClaim]:
        helper_claims = [claim for claim in claims if claim.helper_name]
        if not helper_claims:
            return []

        def _batch_priority(claim: HelperClaim) -> tuple:
            effect = abs(claim.helper_effect) if claim.helper_effect is not None else 0.0
            q_val = claim.helper_q_value if claim.helper_q_value is not None else 1.0
            confidence_rank = {"high": 2, "medium": 1, "low": 0}
            return (-effect, q_val, -confidence_rank.get(claim.confidence.lower(), 0))

        helper_claims.sort(key=_batch_priority)
        cap = self.LINKER_BATCH_HELPER_CAP
        if cap and cap > 0:
            return helper_claims[:cap]
        return helper_claims

    def _assign_missing_theme_ids(
        self,
        claims: List[HelperClaim],
        theme_batch: List[ThemeSummary],
    ) -> None:
        if not claims or not theme_batch:
            return
        themes = [theme for theme in theme_batch if theme is not None]
        if not themes:
            return
        theme_token_cache = {theme.theme_id: self._theme_tokens(theme) for theme in themes}
        for claim in claims:
            if claim.theme_id:
                continue
            helper_tokens = self._helper_tokens(claim)
            best_theme = None
            best_score = -1
            for theme in themes:
                theme_tokens = theme_token_cache.get(theme.theme_id, set())
                overlap = len(helper_tokens & theme_tokens)
                if overlap > best_score:
                    best_score = overlap
                    best_theme = theme
            if best_theme is None:
                best_theme = themes[0]
            claim.theme_id = best_theme.theme_id
            claim.theme_label = best_theme.label

    def _select_background_theme(
        self,
        theme_batch: List[ThemeSummary],
        background_tokens: Set[str],
    ) -> ThemeSummary | None:
        candidates = [theme for theme in theme_batch if theme]
        if not candidates:
            return None
        prioritized = self._prioritize_by_background(candidates, background_tokens)
        return prioritized[0] if prioritized else candidates[0]

    def _prioritize_helper_claims(self, claims: List[HelperClaim]) -> List[HelperClaim]:
        if not claims:
            return claims
        per_direction: Dict[str, List[HelperClaim]] = {"UP": [], "DOWN": []}
        for claim in claims:
            per_direction.setdefault(claim.direction, []).append(claim)
        pruned: List[HelperClaim] = []
        for direction in ("UP", "DOWN"):
            bucket = per_direction.get(direction, [])
            helper_claims = [claim for claim in bucket if claim.helper_name]
            theme_only_claims = [claim for claim in bucket if not claim.helper_name]
            helper_claims.sort(
                key=lambda c: (
                    -(abs(c.helper_effect or 0.0)),
                    c.helper_q_value or 1.0,
                )
            )
            if helper_claims:
                pruned.extend(helper_claims)
            theme_cap = (
                self.THEME_ONLY_CAP_WITH_HELPERS if helper_claims else self.THEME_ONLY_CAP_NO_HELPERS
            )
            theme_only_claims.sort(key=lambda c: c.theme_label)
            pruned.extend(theme_only_claims[:theme_cap])
        if len(pruned) <= self.GLOBAL_CLAIM_CAP:
            return pruned
        pruned.sort(
            key=lambda c: (
                0 if c.helper_name else 1,
                -(abs(c.helper_effect or 0.0)),
                c.helper_q_value or 1.0,
            )
        )
        if self.GLOBAL_CLAIM_CAP and self.GLOBAL_CLAIM_CAP > 0:
            return pruned[: self.GLOBAL_CLAIM_CAP]
        return pruned

    # ------------------------------------------------------------------
    # Evidence (head 2) + Critic (head 3)
    # ------------------------------------------------------------------

    def _gather_evidence(
        self,
        claims: List[HelperClaim],
        themes: Dict[str, List[ThemeSummary]],
        helpers: List[HelperRecord],
        background: Background,
        helpers_available: bool,
    ) -> tuple[Dict[str, EvidenceBundle], List[dict]]:
        evidence_map: Dict[str, EvidenceBundle] = {}
        raw_entries: List[dict] = []
        if not claims:
            return evidence_map, raw_entries

        theme_index = {
            theme.theme_id: theme
            for direction in themes.values()
            for theme in direction
        }
        helper_map = {
            (helper.helper_name, helper.direction): helper for helper in helpers
            if helper.helper_name
        }

        payload_claims = []
        for idx, claim in enumerate(claims):
            if claim.claim_id is None:
                claim.claim_id = f"claim_{idx}"
            theme = theme_index.get(claim.theme_id)
            helper_meta = helper_map.get((claim.helper_name, claim.direction)) if claim.helper_name else None
            payload_claims.append(
                {
                    "claim_id": claim.claim_id,
                    "theme": self._theme_to_dict(theme) if theme else {"theme_id": claim.theme_id, "label": claim.theme_label},
                    "helper": (
                        {
                            "helper_name": helper_meta.helper_name,
                            "helper_class": helper_meta.helper_class,
                            "tf_family": helper_meta.tf_family,
                            "nes": helper_meta.nes,
                            "q_value": helper_meta.q_value,
                            "size": helper_meta.size,
                            "top_hallmark": helper_meta.top_hallmark,
                        }
                        if helper_meta
                        else None
                    ),
                    "claim": {
                        "helper_name": claim.helper_name,
                        "helper_class": claim.helper_class,
                        "direction": claim.direction,
                        "function_phrases": list(claim.function_phrases),
                        "rationale": claim.rationale,
                        "confidence": claim.confidence,
                    },
                }
            )

        base_payload = {
            "background_summary": background.summary,
            "background_fields": background.as_dict(),
            "helpers_available": helpers_available,
        }

        def _call_evidence_batch(claim_batch: List[dict]) -> List[dict]:
            payload = dict(base_payload)
            payload["claims"] = claim_batch
            batch_ids = [entry.get("claim_id") for entry in claim_batch]
            details = {"claim_ids": batch_ids}

            def _invoke() -> Tuple[Any, Optional[str]]:
                response = self.llm(
                    GLOBAL_SYSTEM_PROMPT,
                    f"{EVIDENCE_RETRIEVAL_PROMPT}\nInput JSON:\n{json.dumps(payload)}",
                )
                raw_text = response.content if isinstance(response.content, str) else None
                parsed_json = ensure_json(response)
                return parsed_json, raw_text

            result, _, last_error = self._call_llm_with_retry(
                "evidence_head",
                _invoke,
                details=details,
            )
            if result is None:
                raw_entries.append(
                    {
                        "claim_ids": batch_ids,
                        "source": "llm_error",
                        "used_fallback_snippets": True,
                        "error": last_error,
                    }
                )
                return []

            parsed_json, raw_response_text = result
            entries: List[dict] = []
            if isinstance(parsed_json, list):
                entries = parsed_json
            elif isinstance(parsed_json, dict):
                if parsed_json.get("claim_id"):
                    entries = [parsed_json]
                else:
                    for key in ("claims", "entries", "results", "data"):
                        value = parsed_json.get(key)
                        if isinstance(value, list):
                            entries = value
                            break
            if not entries:
                raw_entries.append(
                    {
                        "claim_ids": batch_ids,
                        "source": "llm_empty",
                        "used_fallback_snippets": True,
                        "raw_response": raw_response_text,
                    }
                )
            return entries

        def _collect_snippets(value) -> List[str]:
            snippets_acc: List[str] = []
            def _extend(val):
                if isinstance(val, str):
                    snippet = val.strip()
                    if snippet:
                        snippets_acc.append(snippet)
                elif isinstance(val, list):
                    for item in val:
                        _extend(item)
                elif isinstance(val, dict):
                    for key in ("evidence", "snippets", "sentences", "items", "values", "text"):
                        if key in val:
                            _extend(val[key])
            _extend(value)
            return snippets_acc

        batch_size = max(1, self.EVIDENCE_BATCH_SIZE)
        for start in range(0, len(payload_claims), batch_size):
            claim_batch = payload_claims[start : start + batch_size]
            parsed_entries = _call_evidence_batch(claim_batch)
            for entry in parsed_entries:
                cid = entry.get("claim_id")
                claim = next((c for c in claims if c.claim_id == cid), None)
                if claim is None:
                    continue
                snippets_list: List[str] = []
                for key in ("evidence", "evidence_snippets", "snippets"):
                    if key in entry:
                        snippets_list.extend(_collect_snippets(entry.get(key)))
                snippets = tuple(snippets_list)
                used_fallback = False
                if not snippets:
                    theme = theme_index.get(claim.theme_id)
                    helper_meta = helper_map.get((claim.helper_name, claim.direction)) if claim.helper_name else None
                    snippets = self._fallback_snippets(claim, theme, helper_meta)
                    used_fallback = True
                alternative = entry.get("alternative") or "Consider alternate regulatory programs."
                gaps = entry.get("missing_or_conflicting") or ""
                predictions = entry.get("predictions")
                if isinstance(predictions, str):
                    predictions_tuple = (predictions.strip(),)
                elif isinstance(predictions, list):
                    predictions_tuple = tuple(str(p).strip() for p in predictions if str(p).strip())
                else:
                    predictions_tuple = tuple()
                if len(predictions_tuple) < 2:
                    predictions_tuple = predictions_tuple + ("Replicate enhancer validation", "Profile driver genes")[: 2 - len(predictions_tuple)]
                bundle = EvidenceBundle(
                    claim=claim,
                    evidence_snippets=snippets,
                    alternative=alternative,
                    gaps=gaps,
                    predictions=predictions_tuple,
                    raw_entry=entry,
                )
                evidence_map[cid] = bundle
                raw_entries.append(
                    self._serialize_evidence_bundle(
                        bundle=bundle,
                        source="llm",
                        used_fallback_snippets=used_fallback,
                    )
                )

        for claim in claims:
            if claim.claim_id not in evidence_map:
                theme = theme_index.get(claim.theme_id)
                bundle = self._default_evidence_bundle(claim, theme)
                evidence_map[claim.claim_id] = bundle
                raw_entries.append(
                    self._serialize_evidence_bundle(
                        bundle=bundle,
                        source="fallback",
                        used_fallback_snippets=True,
                    )
                )

        return evidence_map, raw_entries

    def _critique_claims(
        self,
        claims: List[HelperClaim],
        evidence_map: Dict[str, EvidenceBundle],
        background: Background,
        helpers_available: bool,
    ) -> List[ClaimEvidence]:
        if not claims:
            return []

        payload_claims = []
        for claim in claims:
            bundle = evidence_map.get(claim.claim_id, self._default_evidence_bundle(claim, None))
            helper_meta = {
                "helper_name": claim.helper_name,
                "helper_class": claim.helper_class,
                "helper_effect": claim.helper_effect,
                "helper_q_value": claim.helper_q_value,
            }
            payload_claims.append(
                {
                    "claim_id": claim.claim_id,
                    "theme_id": claim.theme_id,
                    "helper_name": claim.helper_name,
                    "direction": claim.direction,
                    "confidence": claim.confidence,
                    "helper_metadata": helper_meta,
                    "evidence": list(bundle.evidence_snippets),
                    "alternative": bundle.alternative,
                    "missing_or_conflicting": bundle.gaps,
                    "predictions": list(bundle.predictions),
                    "helpers_available": helpers_available,
                }
            )

        payload = {
            "background_summary": background.summary,
            "background_fields": background.as_dict(),
            "claims": payload_claims,
        }

        parsed_entries: List[dict] = []
        target_llm = self.critic_llm or self.llm
        chunk_size = max(1, self.CRITIC_CHUNK_SIZE)
        claim_id_strings = [claim.claim_id for claim in claims if claim.claim_id]

        for chunk_index, claim_chunk in enumerate(self._chunk_sequence(payload_claims, chunk_size), start=1):
            chunk_payload = dict(payload)
            chunk_payload["claims"] = claim_chunk
            details = {
                "claim_count": len(payload_claims),
                "chunk_index": chunk_index,
                "chunk_size": len(claim_chunk),
                "claim_ids": claim_id_strings,
            }

            def _invoke_chunk() -> Any:
                response = target_llm(
                    GLOBAL_SYSTEM_PROMPT,
                    f"{CRITIC_PROMPT}\nInput JSON:\n{json.dumps(chunk_payload)}",
                )
                return ensure_json(response)

            parsed_json, _, _ = self._call_llm_with_retry(
                "critic_head",
                _invoke_chunk,
                details=details,
            )
            if isinstance(parsed_json, list):
                parsed_entries.extend(parsed_json)

        verdict_map = {entry.get("claim_id"): entry for entry in parsed_entries if entry.get("claim_id")}
        fallback_map = {}
        for entry in parsed_entries:
            if entry.get("claim_id"):
                continue
            fallback_map[(entry.get("theme_id"), entry.get("helper_name"))] = entry
        results: List[ClaimEvidence] = []
        for claim in claims:
            bundle = evidence_map.get(claim.claim_id, self._default_evidence_bundle(claim, None))
            critic_entry = verdict_map.get(claim.claim_id) if claim.claim_id else None
            if critic_entry is None:
                critic_entry = fallback_map.get((claim.theme_id, claim.helper_name))
            verdict = critic_entry.get("verdict") if critic_entry and critic_entry.get("verdict") else None
            if not verdict:
                verdict, verdict_reason = self._deterministic_verdict(bundle, claim)
            else:
                verdict_reason = critic_entry.get("verdict_reason") if critic_entry and critic_entry.get("verdict_reason") else "Critic unavailable; treat cautiously."
            alternative = critic_entry.get("alternative") if critic_entry and critic_entry.get("alternative") else bundle.alternative
            gaps = critic_entry.get("gaps") if critic_entry and critic_entry.get("gaps") else bundle.gaps
            predictions_raw = critic_entry.get("predictions") if critic_entry and critic_entry.get("predictions") else bundle.predictions
            if isinstance(predictions_raw, str):
                predictions_tuple = (predictions_raw.strip(),)
            else:
                predictions_tuple = tuple(str(p).strip() for p in predictions_raw) if predictions_raw else bundle.predictions
            results.append(
                ClaimEvidence(
                    claim=claim,
                    evidence_snippets=bundle.evidence_snippets,
                    alternative=alternative,
                    gaps=gaps,
                    predictions=predictions_tuple,
                    verdict=verdict,
                    verdict_reason=verdict_reason,
                )
            )
        return results

    def _deterministic_verdict(self, bundle: EvidenceBundle, claim: HelperClaim) -> tuple[str, str]:
        snippets = tuple(sn for sn in bundle.evidence_snippets if sn.strip())
        conflict = bool(bundle.gaps and "conflict" in bundle.gaps.lower())
        if conflict:
            verdict = "Hypothesis"
            reason = bundle.gaps
        elif len(snippets) >= 2:
            verdict = "Supported"
            reason = "Two or more specific evidence snippets available."
        elif len(snippets) == 1:
            verdict = "Partial"
            reason = "Single evidence snippet available."
        else:
            verdict = "Hypothesis" if claim.helper_class == "theme_only" else "Hypothesis"
            reason = "No evidence snippets provided."
        return verdict, reason

    def _update_theme_helper_effects(
        self,
        themes: Dict[str, List[ThemeSummary]],
        claims: List[HelperClaim],
    ) -> None:
        effect_map: Dict[str, List[float]] = {}
        for claim in claims:
            if claim.helper_name and claim.helper_effect is not None:
                effect_map.setdefault(claim.theme_id, []).append(claim.helper_effect)
        for theme_list in themes.values():
            for theme in theme_list:
                values = effect_map.get(theme.theme_id)
                theme.helper_mean_effect = sum(values) / len(values) if values else None

    def _fallback_evidence(
        self,
        claims: List[HelperClaim],
        theme_index: Dict[str, ThemeSummary],
        helpers_available: bool,
    ) -> List[ClaimEvidence]:
        evidence: List[ClaimEvidence] = []
        for claim in claims:
            theme = theme_index.get(claim.theme_id)
            summary = f"{theme.label if theme else claim.theme_label} shows {claim.direction} shift consistent with {', '.join(claim.function_phrases) or 'pathway activity'}."
            verdict = "Supported" if claim.confidence == "high" else "Partial"
            if not helpers_available or claim.helper_class == "theme_only":
                verdict = "Hypothesis"
            evidence.append(
                ClaimEvidence(
                    claim=claim,
                    evidence_snippets=(summary,),
                    alternative="Consider alternate regulators with similar pathway signatures.",
                    gaps="Needs enhancer validation" if not helpers_available else "",
                    predictions=("Expect enhancer marks at hallmark genes",),
                    verdict=verdict,
                    verdict_reason="Deterministic fallback verdict",
                )
            )
        return evidence

    def _default_evidence_bundle(self, claim: HelperClaim, theme: ThemeSummary | None) -> EvidenceBundle:
        helper_meta = None
        predictions = (
            "Profile enhancer marks at key leading-edge genes",
            "Cross-check helper program in matched atlas",
        )
        return EvidenceBundle(
            claim=claim,
            evidence_snippets=self._fallback_snippets(claim, theme, helper_meta),
            alternative="Consider alternate regulators if enhancer data remain weak.",
            gaps="",
            predictions=predictions,
            raw_entry=None,
        )

    def _serialize_evidence_bundle(
        self,
        bundle: EvidenceBundle,
        source: str,
        used_fallback_snippets: bool,
    ) -> dict:
        claim = bundle.claim
        return {
            "claim_id": claim.claim_id,
            "theme_id": claim.theme_id,
            "theme_label": claim.theme_label,
            "helper_name": claim.helper_name,
            "helper_class": claim.helper_class,
            "direction": claim.direction,
            "helper_top_hallmark": claim.helper_top_hallmark,
            "helper_effect": claim.helper_effect,
            "helper_q_value": claim.helper_q_value,
            "source": source,
            "used_fallback_snippets": used_fallback_snippets,
            "evidence_snippets": list(bundle.evidence_snippets),
            "alternative": bundle.alternative,
            "gaps": bundle.gaps,
            "predictions": list(bundle.predictions),
            "raw_entry": bundle.raw_entry,
        }

    def _fallback_snippets(
        self,
        claim: HelperClaim,
        theme: ThemeSummary | None,
        helper_meta: HelperRecord | None,
    ) -> tuple[str, ...]:
        snippets = []
        if theme:
            pathways = ", ".join(tp.term for tp in theme.top_pathways[:2]) if theme.top_pathways else "theme pathways"
            genes = ", ".join(theme.leading_edges[:3]) if theme.leading_edges else "leading-edge genes"
            snippets.append(f"{theme.label} {claim.direction} shift tracks {pathways} and genes {genes}.")
        else:
            snippets.append(f"{claim.theme_label} {claim.direction} shift follows its leading-edge genes.")
        if claim.helper_name:
            if helper_meta:
                hallmark = helper_meta.top_hallmark or helper_meta.helper_class
                snippets.append(f"Helper {helper_meta.helper_name} reflects {hallmark} activity supporting this response.")
            else:
                mark = claim.helper_top_hallmark or claim.helper_class
                snippets.append(f"Helper {claim.helper_name} captures {mark} program consistent with the theme.")
        return tuple(snippets)


    THEME_ONLY_FALLBACK_CAP = 3

    def _ensure_background_theme_claims(
        self,
        claims: List[HelperClaim],
        candidates: Dict[str, List[ThemeSummary]],
        background: Background,
    ) -> List[HelperClaim]:
        if not candidates:
            return claims
        tokens = self._collect_background_tokens(background)
        if not tokens:
            return claims
        existing = {claim.theme_id for claim in claims if claim.theme_id}
        for direction in ("UP", "DOWN"):
            ordered_themes = self._prioritize_by_background(candidates.get(direction, []), tokens)
            for theme in ordered_themes:
                if theme.theme_id in existing:
                    continue
                slug_tokens = [tok for tok in _slugify_text(theme.label).split("-") if tok]
                label_tokens = [tok for tok in re.findall(r"[a-z0-9]+", theme.label.lower()) if len(tok) >= 3]
                keywords = [tok for tok in slug_tokens + label_tokens if tok not in BACKGROUND_STOPWORDS]
                if not keywords:
                    continue
                if any(keyword in tokens for keyword in keywords):
                    claims.insert(0, self._make_background_theme_claim(theme))
                    existing.add(theme.theme_id)
        return claims

    def _ensure_direction_coverage(
        self,
        claims: List[HelperClaim],
        candidates: Dict[str, List[ThemeSummary]],
    ) -> List[HelperClaim]:
        if not candidates:
            return claims
        existing_ids = {claim.theme_id for claim in claims if claim.theme_id}
        for direction in ("UP", "DOWN"):
            helper_count = sum(
                1 for claim in claims if claim.direction == direction and claim.helper_name
            )
            if helper_count > 0:
                continue
            added = 0
            for theme in candidates.get(direction, []):
                if theme.theme_id in existing_ids:
                    continue
                claims.append(self._make_background_theme_claim(theme))
                existing_ids.add(theme.theme_id)
                added += 1
                if added >= self.THEME_ONLY_FALLBACK_CAP:
                    break
        return claims

    def _retarget_helper_claims(
        self,
        claims: List[HelperClaim],
        candidates: Dict[str, List[ThemeSummary]],
    ) -> List[HelperClaim]:
        return claims

    def _make_background_theme_claim(self, theme: ThemeSummary) -> HelperClaim:
        rationale = (
            f"{theme.label} theme is explicitly requested in the study background; "
            "retaining it as a hypothesis until enhancer helpers confirm the signal."
        )
        return HelperClaim(
            theme_id=theme.theme_id,
            theme_label=theme.label,
            helper_name=None,
            helper_class="theme_only",
            direction=theme.direction,
            function_phrases=(theme.label,),
            rationale=rationale,
            confidence="medium",
        )

    def _collect_background_tokens(self, background: Background) -> Set[str]:
        corpus_parts = [background.summary.lower()] if background and background.summary else []
        corpus_parts.extend(value.lower() for value in background.as_dict().values())
        corpus = " ".join(corpus_parts)
        if not corpus.strip():
            return set()
        tokens = {token for token in re.findall(r"[a-z0-9]+", corpus) if len(token) >= 3}
        tokens -= BACKGROUND_STOPWORDS
        return tokens

    def _prioritize_by_background(
        self,
        theme_list: List[ThemeSummary] | None,
        tokens: Set[str],
    ) -> List[ThemeSummary]:
        if not theme_list:
            return []
        prioritized: List[ThemeSummary] = []
        fallback: List[ThemeSummary] = []
        for theme in theme_list:
            slug_tokens = {tok for tok in _slugify_text(theme.label).split("-") if tok}
            label_tokens = {tok for tok in re.findall(r"[a-z0-9]+", theme.label.lower()) if len(tok) >= 3}
            keywords = {tok for tok in slug_tokens | label_tokens if tok not in BACKGROUND_STOPWORDS}
            if tokens and keywords & tokens:
                prioritized.append(theme)
            else:
                fallback.append(theme)
        return prioritized + fallback

    def _select_themes_from_candidates(
        self,
        candidates: Dict[str, List[ThemeSummary]],
        claims: List[HelperClaim],
        priority_tokens: Set[str],
        helpers: List[HelperRecord],
    ) -> tuple[List[HelperClaim], Dict[str, List[ThemeSummary]]]:
        score_map = self._compute_theme_semantic_scores(candidates, claims, priority_tokens, helpers)
        candidate_lookup: Dict[str, ThemeSummary] = {}
        candidate_order: Dict[str, List[str]] = {"UP": [], "DOWN": []}
        for direction in ("UP", "DOWN"):
            ordered = sorted(
                candidates.get(direction, []),
                key=lambda theme: (
                    -score_map.get(theme.theme_id, 0.0),
                    -abs(theme.effect),
                ),
            )
            for theme in ordered:
                candidate_lookup[theme.theme_id] = theme
                candidate_order.setdefault(direction, []).append(theme.theme_id)

        helper_ids: Dict[str, List[str]] = {"UP": [], "DOWN": []}
        theme_only_ids: Dict[str, List[str]] = {"UP": [], "DOWN": []}
        for claim in claims:
            theme = candidate_lookup.get(claim.theme_id)
            if theme is None:
                continue
            bucket = helper_ids if claim.helper_name else theme_only_ids
            if claim.theme_id not in bucket[theme.direction]:
                bucket[theme.direction].append(claim.theme_id)

        final_theme_ids: Dict[str, List[str]] = {"UP": [], "DOWN": []}
        for direction in ("UP", "DOWN"):
            order = candidate_order.get(direction, [])
            ids: List[str] = []
            ids.extend(helper_ids.get(direction, []))
            remaining_slots = max(0, self.analysis.theme_cap - len(ids)) if self.analysis.theme_cap > 0 else None
            theme_only_cap = 3
            for theme_id in theme_only_ids.get(direction, []):
                if remaining_slots is not None and remaining_slots <= 0:
                    break
                if theme_only_cap <= 0:
                    break
                ids.append(theme_id)
                if remaining_slots is not None:
                    remaining_slots -= 1
                theme_only_cap -= 1
            if remaining_slots is not None:
                for theme_id in order:
                    if theme_id not in ids:
                        ids.append(theme_id)
                        remaining_slots -= 1
                    if remaining_slots <= 0:
                        break
            final_theme_ids[direction] = ids

        for direction in ("UP", "DOWN"):
            priority_theme = next(
                (
                    theme_id
                    for theme_id in candidate_order.get(direction, [])
                    if self._theme_tokens(candidate_lookup[theme_id]) & priority_tokens
                ),
                None,
            )
            if priority_theme and priority_theme not in final_theme_ids[direction]:
                final_theme_ids[direction].insert(0, priority_theme)
            elif priority_theme and final_theme_ids[direction]:
                reordered = [priority_theme] + [theme_id for theme_id in final_theme_ids[direction] if theme_id != priority_theme]
                final_theme_ids[direction] = reordered

        filtered_claims = [
            claim for claim in claims if claim.theme_id in candidate_lookup and claim.theme_id in final_theme_ids.get(candidate_lookup[claim.theme_id].direction, [])
        ]

        selected_themes: Dict[str, List[ThemeSummary]] = {"UP": [], "DOWN": []}
        for direction in ("UP", "DOWN"):
            for theme_id in final_theme_ids.get(direction, []):
                theme = candidate_lookup.get(theme_id)
                if theme:
                    selected_themes[direction].append(theme)

        supported_themes = {claim.theme_id for claim in claims if claim.helper_name}
        total_cap = max(len(supported_themes), self.analysis.theme_cap_total, 1)
        while (len(selected_themes["UP"]) + len(selected_themes["DOWN"])) > total_cap:
            removable: List[tuple[str, int, ThemeSummary]] = []
            for direction in ("UP", "DOWN"):
                for idx, theme in enumerate(selected_themes[direction]):
                    removable.append((direction, idx, theme))
            if not removable:
                break
            removable.sort(
                key=lambda item: (
                    0 if item[2].theme_id not in supported_themes else 1,
                    abs(item[2].effect),
                )
            )
            direction, remove_idx, dropped_theme = removable[0]
            selected_themes[direction].pop(remove_idx)
            filtered_claims = [claim for claim in filtered_claims if claim.theme_id != dropped_theme.theme_id]
        return filtered_claims, selected_themes

    def _compute_theme_semantic_scores(
        self,
        candidates: Dict[str, List[ThemeSummary]],
        claims: List[HelperClaim],
        priority_tokens: Set[str],
        helpers: List[HelperRecord],
    ) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        claims_by_theme: Dict[str, List[HelperClaim]] = {}
        for claim in claims:
            if claim.theme_id:
                claims_by_theme.setdefault(claim.theme_id, []).append(claim)
        helper_tokens_cache = {
            helper.helper_name: self._tokenize_text(
                helper.helper_name or "",
                helper.helper_class or "",
                helper.top_hallmark or "",
            )
            for helper in helpers
            if helper.helper_name
        }
        for theme_list in candidates.values():
            for theme in theme_list:
                theme_tokens = self._theme_tokens(theme)
                score = 0.0
                if priority_tokens & theme_tokens:
                    score += 10.0
                for claim in claims_by_theme.get(theme.theme_id, []):
                    helper_tokens = self._helper_tokens(claim)
                    overlap = len(theme_tokens & helper_tokens)
                    if claim.helper_name and overlap:
                        score += 3.0 + overlap * 1.5
                    elif claim.helper_name:
                        score += 1.0
                    elif priority_tokens & theme_tokens:
                        score += 2.5
                for helper_name, tokens in helper_tokens_cache.items():
                    overlap = len(theme_tokens & tokens)
                    if overlap:
                        score += 3.0 * overlap
                score += abs(theme.effect) * 0.1
                scores[theme.theme_id] = score
        return scores

    def _helper_tokens(self, claim: HelperClaim) -> Set[str]:
        tokens = self._tokenize_text(
            claim.helper_name or "",
            claim.helper_class or "",
            claim.helper_top_hallmark or "",
        )
        return tokens


    def _theme_tokens(self, theme: ThemeSummary) -> Set[str]:
        tokens = self._tokenize_text(theme.label, theme.collection or "")
        slug = _slugify_text(theme.label)
        tokens |= {tok for tok in slug.split("-") if tok and tok not in BACKGROUND_STOPWORDS}
        for record in theme.top_pathways:
            term = getattr(record, "term", None)
            if term:
                tokens |= self._tokenize_text(term)
        for gene in theme.leading_edges:
            tokens.add(gene.lower())
        if "emt" in tokens or "adhesion" in tokens or "ecm" in tokens:
            tokens.update({"mesenchymal", "mesenchyme"})
        tokens -= BACKGROUND_STOPWORDS
        return tokens

    @staticmethod
    def _tokenize_text(*parts: str) -> Set[str]:
        tokens: Set[str] = set()
        for part in parts:
            if not part:
                continue
            for token in re.findall(r"[a-z0-9]+", part.lower()):
                if len(token) >= 3:
                    tokens.add(token)
        return tokens

    def _dedupe_claims(self, claims: List[HelperClaim]) -> List[HelperClaim]:
        if not claims:
            return claims

        confidence_rank = {"high": 3, "medium": 2, "low": 1}

        def _priority(claim: HelperClaim) -> tuple:
            conf_score = confidence_rank.get(claim.confidence.lower(), 0)
            helper_effect = abs(claim.helper_effect) if claim.helper_effect is not None else 0.0
            rationale_len = len(claim.rationale or "")
            return (conf_score, helper_effect, rationale_len)

        best_by_key: Dict[tuple, HelperClaim] = {}
        first_seen_index: Dict[tuple, int] = {}

        for idx, claim in enumerate(claims):
            key = (claim.theme_id, claim.helper_name)
            if claim.helper_name is None:
                best_by_key[(claim.theme_id, f"theme_only_{idx}")] = claim
                first_seen_index[(claim.theme_id, f"theme_only_{idx}")] = idx
                continue
            if key not in best_by_key:
                best_by_key[key] = claim
                first_seen_index[key] = idx
                continue
            if _priority(claim) > _priority(best_by_key[key]):
                best_by_key[key] = claim

        sorted_items = sorted(first_seen_index.items(), key=lambda item: item[1])
        result: List[HelperClaim] = []
        for key, _ in sorted_items:
            claim = best_by_key.get(key)
            if claim:
                result.append(claim)
        return result

    def _dedupe_claims_with_evidence(
        self,
        claims: List[HelperClaim],
        evidence: List[ClaimEvidence],
    ) -> tuple[List[HelperClaim], List[ClaimEvidence]]:
        if not claims or not evidence or len(claims) != len(evidence):
            return claims, evidence
        seen: Set[tuple[str, str | None]] = set()
        filtered_claims: List[HelperClaim] = []
        filtered_evidence: List[ClaimEvidence] = []
        for claim, ev in zip(claims, evidence):
            key = (claim.theme_id, claim.helper_name)
            if key in seen:
                continue
            seen.add(key)
            filtered_claims.append(claim)
            filtered_evidence.append(ev)
        return filtered_claims, filtered_evidence

    # ------------------------------------------------------------------
    # Micro revision loop (risk-gated)
    # ------------------------------------------------------------------

    def _revise_claims(
        self,
        claims: List[HelperClaim],
        claim_evidence: List[ClaimEvidence],
        themes: Dict[str, List[ThemeSummary]],
        helpers: List[HelperRecord],
        background: Background,
        helpers_available: bool,
    ) -> tuple[List[HelperClaim], List[ClaimEvidence], List[dict]]:
        if not claims:
            return claims, claim_evidence, []

        theme_index = {
            theme.theme_id: theme
            for direction in themes.values()
            for theme in direction
        }
        theme_order = [
            theme.theme_id
            for direction in ("UP", "DOWN")
            for theme in themes.get(direction, [])
        ]
        helper_lookup = {
            (record.helper_name, record.direction): record
            for record in helpers
            if record.helper_name
        }
        helpers_by_direction: Dict[str, List[HelperRecord]] = {"UP": [], "DOWN": []}
        for record in helpers:
            helpers_by_direction.setdefault(record.direction, []).append(record)
        for bucket in helpers_by_direction.values():
            bucket.sort(key=lambda rec: (rec.q_value, -abs(rec.nes)))

        duplicate_flags = self._identify_duplicate_claims(claims)
        background_fields = background.as_dict()

        evidence_map = {}
        for ev in claim_evidence:
            key = ev.claim.claim_id or id(ev.claim)
            evidence_map[key] = ev

        used_helpers: Dict[str, set[str]] = {}
        final_claims: List[HelperClaim] = []
        final_evidence: List[ClaimEvidence] = []
        revision_notes: List[dict] = []

        for claim in claims:
            ev = evidence_map.get(claim.claim_id or id(claim))
            if ev is None:
                continue

            helper_ref = (
                helper_lookup.get((claim.helper_name, claim.direction))
                if claim.helper_name
                else None
            )
            theme = theme_index.get(claim.theme_id)
            issues = self._collect_revision_gates(
                claim,
                ev,
                helper_ref,
                duplicate_flags,
            )
            if not issues:
                calibrated_claim = self._calibrate_claim_confidence(claim, ev)
                updated_ev = self._clone_evidence(ev, calibrated_claim)
                final_claims.append(calibrated_claim)
                final_evidence.append(updated_ev)
                if calibrated_claim.helper_name:
                    used_helpers.setdefault(claim.theme_id, set()).add(calibrated_claim.helper_name)
                continue

            revised_claim, revised_ev, note = self._apply_revision_actions(
                claim=claim,
                evidence=ev,
                issues=issues,
                theme=theme,
                helper_ref=helper_ref,
                helpers_by_direction=helpers_by_direction,
                used_helpers=used_helpers.setdefault(claim.theme_id, set()),
                background_fields=background_fields,
                helpers_available=helpers_available,
            )
            if revised_claim:
                final_claims.append(revised_claim)
                final_evidence.append(revised_ev)
                if revised_claim.helper_name:
                    used_helpers.setdefault(revised_claim.theme_id, set()).add(revised_claim.helper_name)
            if note:
                revision_notes.append(note)

        self._validate_direction_rules(final_claims, helper_lookup)
        return final_claims, final_evidence, revision_notes

    def _identify_duplicate_claims(self, claims: List[HelperClaim]) -> Dict[str, bool]:
        duplicates: Dict[str, bool] = {}
        seen: Dict[str, str] = {}
        for claim in claims:
            if not claim.helper_name:
                continue
            signature = self._helper_signature(claim)
            if signature in seen:
                duplicates[signature] = True
            else:
                seen[signature] = claim.theme_id
        return duplicates

    def _helper_signature(self, claim: HelperClaim) -> str:
        label = (claim.helper_name or "").lower()
        label = "".join(ch for ch in label if ch.isalnum())
        hallmark = (claim.helper_top_hallmark or "").lower()
        return f"{claim.theme_id}|{claim.helper_class}|{hallmark or label}"

    def _collect_revision_gates(
        self,
        claim: HelperClaim,
        evidence: ClaimEvidence,
        helper_ref: HelperRecord | None,
        duplicate_flags: Dict[str, bool],
    ) -> List[str]:
        issues: List[str] = []
        verdict = evidence.verdict.lower()
        if helper_ref and helper_ref.direction != claim.direction:
            issues.append("direction mismatch")
        if verdict == "not supported":
            issues.append("weak evidence")
        elif verdict == "hypothesis" and claim.helper_class != "theme_only":
            issues.append("weak evidence")
        elif verdict == "partial" and len(evidence.evidence_snippets) < 2:
            issues.append("weak evidence")
        if len(evidence.evidence_snippets) < 2:
            issues.append("insufficient evidence")
        text_blocks = [
            evidence.gaps or "",
            evidence.alternative or "",
        ]
        if any("contrad" in block.lower() or "opposite" in block.lower() for block in text_blocks if block):
            issues.append("background conflict")
        return issues

    def _apply_revision_actions(
        self,
        claim: HelperClaim,
        evidence: ClaimEvidence,
        issues: List[str],
        theme: ThemeSummary | None,
        helper_ref: HelperRecord | None,
        helpers_by_direction: Dict[str, List[HelperRecord]],
        used_helpers: set[str],
        background_fields: Dict[str, str],
        helpers_available: bool,
    ) -> tuple[HelperClaim | None, ClaimEvidence | None, dict | None]:
        reason_set = set(issues)
        reason = ", ".join(sorted(reason_set))
        severe_issues = {"direction mismatch", "background conflict"}
        if not severe_issues.intersection(reason_set):
            return claim, evidence, None
        prefer_class = None
        if "redundant helper" in issues and claim.helper_class in {"celltype", "tf_family"}:
            prefer_class = "tf_family" if claim.helper_class == "celltype" else "celltype"

        alternative = self._pick_alternative_helper(
            direction=claim.direction,
            helpers_by_direction=helpers_by_direction,
            used_helpers=used_helpers,
            current_name=claim.helper_name,
            prefer_class=prefer_class,
        )

        if alternative:
            revised_claim = self._build_helper_claim(
                theme_id=claim.theme_id,
                theme_label=claim.theme_label,
                theme=theme,
                helper=alternative,
                direction=claim.direction,
                background_fields=background_fields,
            )
            revised_ev = self._build_revision_evidence(
                revised_claim,
                theme,
                helpers_available,
                verdict="Partial",
                reason="Deterministic swap pending enhancer confirmation.",
            )
            note = {
                "theme_id": claim.theme_id,
                "original_helper": claim.helper_name,
                "new_helper": revised_claim.helper_name,
                "action": "replacement",
                "reason": reason,
            }
            return revised_claim, revised_ev, note

        revised_claim = self._build_theme_only_claim(
            theme=theme,
            theme_id=claim.theme_id,
            theme_label=claim.theme_label,
            direction=claim.direction,
            background_fields=background_fields,
        )
        revised_ev = self._build_revision_evidence(
            revised_claim,
            theme,
            helpers_available=False,
            verdict="Hypothesis",
            reason="No helper cleared revision gates; relying on leading-edge genes.",
        )
        note = {
            "theme_id": claim.theme_id,
            "original_helper": claim.helper_name,
            "new_helper": None,
            "action": "downgraded_to_theme_only",
            "reason": reason,
        }
        return revised_claim, revised_ev, note

    def _pick_alternative_helper(
        self,
        direction: str,
        helpers_by_direction: Dict[str, List[HelperRecord]],
        used_helpers: set[str],
        current_name: str | None,
        prefer_class: str | None,
    ) -> HelperRecord | None:
        bucket = helpers_by_direction.get(direction, [])
        excluded = set(used_helpers)
        if current_name:
            excluded.add(current_name)

        def _select(prefer: str | None) -> HelperRecord | None:
            for candidate in bucket:
                if candidate.helper_name in excluded:
                    continue
                if prefer and candidate.helper_class != prefer:
                    continue
                return candidate
            return None

        if prefer_class:
            alt = _select(prefer_class)
            if alt:
                return alt
        return _select(None)

    def _build_helper_claim(
        self,
        theme: ThemeSummary | None,
        theme_id: str,
        theme_label: str,
        helper: HelperRecord,
        direction: str,
        background_fields: Dict[str, str],
    ) -> HelperClaim:
        phrase = self._hallmark_to_phrase(helper.top_hallmark) or f"{helper.helper_class} program"
        leading = ", ".join(list(theme.leading_edges[:3])) if theme and theme.leading_edges else ""
        expected = background_fields.get("Expected_Phenotypes_or_Trends_(optional, describe expectations not mandates)", "")
        rationale_bits = [
            f"{theme.label if theme else 'The pathway'} {direction.lower()} shift is driven by genes such as {leading}" if leading else f"{theme.label if theme else 'The pathway'} follows the observed shift",
            f"{helper.helper_name or helper.helper_class} captures the {phrase} activity linked to this response.",
        ]
        if expected:
            rationale_bits.append(f"Expectation: {expected[:140]}")
        rationale = " ".join(bit for bit in rationale_bits if bit)
        return HelperClaim(
            theme_id=theme_id,
            theme_label=theme.label if theme else theme_label,
            helper_name=helper.helper_name,
            helper_class=helper.helper_class,
            direction=direction,
            function_phrases=(phrase,),
            rationale=rationale,
            confidence="medium" if helper.top_hallmark else "low",
            helper_top_hallmark=helper.top_hallmark,
            helper_effect=helper.nes,
            helper_q_value=helper.q_value,
        )

    def _build_theme_only_claim(
        self,
        theme: ThemeSummary | None,
        theme_id: str,
        theme_label: str,
        direction: str,
        background_fields: Dict[str, str],
    ) -> HelperClaim:
        label = theme.label if theme else theme_label
        leading = ", ".join(list(theme.leading_edges[:3])) if theme and theme.leading_edges else ""
        expected = background_fields.get("Expected_Phenotypes_or_Trends_(optional, describe expectations not mandates)", "")
        rationale = f"{label} {direction.lower()} shift is retained via leading-edge genes ({leading}) even though no enhancer helper survived gating."
        if expected:
            rationale += f" Expected trend: {expected[:140]}."
        return HelperClaim(
            theme_id=theme.theme_id if theme else theme_id,
            theme_label=label,
            helper_name=None,
            helper_class="theme_only",
            direction=direction,
            function_phrases=(f"{label} program",),
            rationale=rationale,
            confidence="low",
            helper_top_hallmark=None,
            helper_effect=None,
            helper_q_value=None,
        )

    def _build_revision_evidence(
        self,
        claim: HelperClaim,
        theme: ThemeSummary | None,
        helpers_available: bool,
        verdict: str,
        reason: str,
    ) -> ClaimEvidence:
        if theme and theme.top_pathways:
            leading_terms = ", ".join(tp.term for tp in theme.top_pathways[:2])
        else:
            leading_terms = claim.theme_label
        snippet1 = (
            f"{theme.label if theme else claim.theme_label} {claim.direction} signal concentrates in sets such as {leading_terms}"
        )
        if claim.helper_name:
            phrase = claim.function_phrases[0] if claim.function_phrases else claim.helper_class
            snippet2 = f"{claim.helper_name} models the {phrase} module consistent with this direction."
        else:
            snippet2 = "No helper cleared QC; interpretation falls back to leading-edge genes and background expectations."
        predictions = (
            f"Check enhancer marks at {', '.join(list(theme.leading_edges[:2]))}" if theme and theme.leading_edges else "Profile key leading-edge genes for enhancer activation",
            "Test whether the highlighted process reverses when perturbation is removed.",
        )
        return ClaimEvidence(
            claim=claim,
            evidence_snippets=(snippet1, snippet2),
            alternative="Await direct enhancer validation or motif rescue experiment.",
            gaps=reason,
            predictions=predictions,
            verdict=verdict,
            verdict_reason=reason,
        )

    def _clone_evidence(self, evidence: ClaimEvidence, new_claim: HelperClaim) -> ClaimEvidence:
        clone = ClaimEvidence(
            claim=new_claim,
            evidence_snippets=evidence.evidence_snippets,
            alternative=evidence.alternative,
            gaps=evidence.gaps,
            predictions=evidence.predictions,
            verdict=evidence.verdict,
            verdict_reason=evidence.verdict_reason,
        )
        return clone

    def _validate_direction_rules(
        self,
        claims: List[HelperClaim],
        helper_lookup: Dict[tuple[str, str], HelperRecord],
    ) -> None:
        for claim in claims:
            if not claim.helper_name:
                continue
            helper = helper_lookup.get((claim.helper_name, claim.direction))
            if helper and helper.direction != claim.direction:
                raise ValueError(
                    f"Direction mismatch after revision: {claim.helper_name} ({helper.direction}) vs {claim.direction}"
                )

    def _calibrate_claim_confidence(self, claim: HelperClaim, evidence: ClaimEvidence) -> HelperClaim:
        verdict = evidence.verdict.lower()
        confidence = claim.confidence
        if verdict == "supported":
            confidence = "high"
        elif verdict == "partial":
            confidence = "medium"
        elif verdict in {"hypothesis", "not supported"}:
            confidence = "low"
        if confidence == claim.confidence:
            return claim
        return HelperClaim(
            theme_id=claim.theme_id,
            theme_label=claim.theme_label,
            helper_name=claim.helper_name,
            helper_class=claim.helper_class,
            direction=claim.direction,
            function_phrases=claim.function_phrases,
            rationale=claim.rationale,
            confidence=confidence,
            helper_top_hallmark=claim.helper_top_hallmark,
             helper_effect=claim.helper_effect,
             helper_q_value=claim.helper_q_value,
             claim_id=claim.claim_id,
        )

    def _hallmark_to_phrase(self, hallmark: str | None) -> str | None:
        if not hallmark:
            return None
        if hallmark.upper().startswith("HALLMARK_"):
            hallmark = hallmark[len("HALLMARK_") :]
        return hallmark.replace("_", " ").title()

    def _trim_thesis_claims(
        self,
        claim_evidence: List[ClaimEvidence],
    ) -> List[ClaimEvidence]:
        # Always return the full claim set so the mini-thesis sees every helper/theme link.
        return list(claim_evidence)
        return selected

    def _select_thesis_themes(
        self,
        themes: Dict[str, List[ThemeSummary]],
        claim_evidence: List[ClaimEvidence],
    ) -> Dict[str, List[ThemeSummary]]:
        selected: Dict[str, List[ThemeSummary]] = {"UP": [], "DOWN": []}
        helper_theme_ids = {
            ev.claim.theme_id for ev in claim_evidence if ev.claim.helper_name
        }
        theme_lookup: Dict[str, ThemeSummary] = {
            theme.theme_id: theme for direction in themes.values() for theme in direction
        }

        for direction in ("UP", "DOWN"):
            direction_themes = themes.get(direction, [])
            helper_block = [theme for theme in direction_themes if theme.theme_id in helper_theme_ids]
            helper_block = helper_block[: self.MINI_THESIS_HELPER_THEME_PRIORITY]
            selected[direction].extend(helper_block)
            remaining_slots = self.MINI_THESIS_THEME_CAP_PER_DIRECTION - len(selected[direction])
            if remaining_slots > 0:
                theme_only_block = [theme for theme in direction_themes if theme.theme_id not in helper_theme_ids]
                theme_only_block.sort(key=lambda item: abs(item.effect), reverse=True)
                selected[direction].extend(theme_only_block[:remaining_slots])

        for ev in claim_evidence:
            theme = theme_lookup.get(ev.claim.theme_id)
            if not theme:
                continue
            bucket = selected.get(theme.direction, [])
            if theme not in bucket:
                bucket.append(theme)

        for direction in ("UP", "DOWN"):
            bucket = selected.get(direction, [])
            if len(bucket) > self.MINI_THESIS_THEME_CAP_PER_DIRECTION:
                selected[direction] = bucket[: self.MINI_THESIS_THEME_CAP_PER_DIRECTION]

        return selected

    # ------------------------------------------------------------------
    # Mini-thesis composer and artefacts
    # ------------------------------------------------------------------

    def _compose_mini_thesis(
        self,
        themes: Dict[str, List[ThemeSummary]],
        claim_evidence: List[ClaimEvidence],
        background: Background,
        helpers_available: bool,
    ) -> str:
        trimmed_claims = self._trim_thesis_claims(claim_evidence)
        thesis_themes = self._select_thesis_themes(themes, trimmed_claims)
        theme_payload = [
            self._theme_to_dict(theme)
            for direction in ("UP", "DOWN")
            for theme in thesis_themes.get(direction, [])
        ]
        evidence_map = {
            (ev.claim.theme_id, ev.claim.helper_name): ev for ev in claim_evidence
        }
        claims_payload = []
        for ev in trimmed_claims:
            claims_payload.append(
                {
                    "theme_id": ev.claim.theme_id,
                    "theme_label": ev.claim.theme_label,
                    "helper_name": ev.claim.helper_name,
                    "helper_class": ev.claim.helper_class,
                    "direction": ev.claim.direction,
                    "function_phrases": list(ev.claim.function_phrases),
                    "rationale": ev.claim.rationale,
                    "confidence": ev.claim.confidence,
                    "verdict": ev.verdict,
                    "verdict_reason": ev.verdict_reason,
                    "evidence_snippets": list(ev.evidence_snippets),
                    "predictions": list(ev.predictions),
                    "alternative": ev.alternative,
                    "gaps": ev.gaps,
                }
            )
        payload = {
            "background_summary": background.summary,
            "background_fields": background.as_dict(),
            "helpers_available": helpers_available,
            "themes": theme_payload,
            "claims": claims_payload,
        }
        details = {
            "theme_total": sum(len(thesis_themes.get(direction, [])) for direction in ("UP", "DOWN")),
            "claim_total": len(trimmed_claims),
        }

        def _invoke() -> str:
            response = self.llm(
                GLOBAL_SYSTEM_PROMPT,
                f"{MINI_THESIS_PROMPT}\nInput JSON:\n{json.dumps(payload)}",
            )
            content = response.content.strip()
            if not content:
                raise LLMError("Mini-thesis response was empty.")
            return content

        thesis_text, _, _ = self._call_llm_with_retry(
            "mini_thesis",
            _invoke,
            details=details,
        )
        if isinstance(thesis_text, str) and thesis_text.strip():
            return thesis_text.strip()

        lines = ["Mini-thesis unavailable; fallback summary:", ""]
        for direction in ("UP", "DOWN"):
            block = themes.get(direction, [])
            if not block:
                continue
            lines.append(f"{direction} programs:")
            for theme in block:
                lines.append(f"- {theme.label}: q={theme.q_value:.3g}, effect={theme.effect:.2f}")
        lines.append("")
        lines.append("Claims:")
        for ev in claim_evidence:
            lines.append(
                f"- {ev.claim.theme_label} ↔ {ev.claim.helper_name or 'theme-only'} "
                f"({ev.verdict}): {ev.claim.rationale}"
            )
        return "\n".join(lines)

    def _write_artifacts(
        self,
        output_dir: Path,
        themes: Dict[str, List[ThemeSummary]],
        helpers: List[HelperRecord],
        claims: List[HelperClaim],
        claim_evidence: List[ClaimEvidence],
        evidence_head_raw: List[dict],
        mini_thesis: str,
        helpers_available: bool,
        revision_notes: List[dict],
        stage_failures: List[dict],
    ) -> None:
        theme_payload = {
            direction: [self._theme_to_dict(theme) for theme in theme_list]
            for direction, theme_list in themes.items()
        }
        helper_payload = [
            {
                "helper_name": helper.helper_name,
                "helper_class": helper.helper_class,
                "tf_family": helper.tf_family,
                "direction": helper.direction,
                "nes": helper.nes,
                "q_value": helper.q_value,
                "size": helper.size,
                "top_hallmark": helper.top_hallmark,
            }
            for helper in helpers
        ]
        claims_payload = [
            {
                "theme_id": claim.theme_id,
                "theme_label": claim.theme_label,
                "helper_name": claim.helper_name,
                "helper_class": claim.helper_class,
                "direction": claim.direction,
                "function_phrases": list(claim.function_phrases),
                "rationale": claim.rationale,
                "confidence": claim.confidence,
                "helper_top_hallmark": claim.helper_top_hallmark,
                "helper_effect": claim.helper_effect,
                "helper_q_value": claim.helper_q_value,
            }
            for claim in claims
        ]
        claim_evidence_payload = [
            {
                "theme_id": ev.claim.theme_id,
                "helper_name": ev.claim.helper_name,
                "verdict": ev.verdict,
                "verdict_reason": ev.verdict_reason,
                "evidence_snippets": list(ev.evidence_snippets),
                "predictions": list(ev.predictions),
                "alternative": ev.alternative,
                "gaps": ev.gaps,
            }
            for ev in claim_evidence
        ]
        (output_dir / "themes_condensed.json").write_text(json.dumps(theme_payload, indent=2), encoding="utf-8")
        (output_dir / "helpers.json").write_text(json.dumps(helper_payload, indent=2), encoding="utf-8")
        (output_dir / "helper_claims.json").write_text(json.dumps(claims_payload, indent=2), encoding="utf-8")
        (output_dir / "claim_reviews.json").write_text(json.dumps(claim_evidence_payload, indent=2), encoding="utf-8")
        (output_dir / "evidence_head_raw.json").write_text(json.dumps(evidence_head_raw, indent=2), encoding="utf-8")
        (output_dir / "stage_failures.json").write_text(json.dumps(stage_failures, indent=2), encoding="utf-8")
        (output_dir / "mini_thesis.txt").write_text(mini_thesis, encoding="utf-8")
        (output_dir / "revision_notes.json").write_text(json.dumps(revision_notes, indent=2), encoding="utf-8")
        with (output_dir / "evidence_box.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "theme_id",
                    "theme_label",
                    "helper_name",
                    "helper_class",
                    "verdict",
                    "verdict_reason",
                    "prediction_1",
                    "prediction_2",
                ]
            )
            for ev in claim_evidence:
                preds = list(ev.predictions)
                preds += [""] * (2 - len(preds))
                writer.writerow(
                    [
                        ev.claim.theme_id,
                        ev.claim.theme_label,
                        ev.claim.helper_name or "theme_only",
                        ev.claim.helper_class,
                        ev.verdict,
                        ev.verdict_reason,
                        preds[0],
                        preds[1],
                    ]
                )
        availability_note = "available" if helpers_available else "missing"
        (output_dir / "helpers_status.txt").write_text(f"Enhancer helper evidence: {availability_note}\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_sequence(items: Sequence[Any], chunk_size: int) -> Iterable[List[Any]]:
        if chunk_size <= 0:
            chunk_size = 1
        for start in range(0, len(items), chunk_size):
            yield list(items[start : start + chunk_size])

    @staticmethod
    def _theme_to_dict(theme: ThemeSummary | None) -> dict:
        if theme is None:
            return {}
        return {
            "theme_id": theme.theme_id,
            "label": theme.label,
            "direction": theme.direction,
            "collection": theme.collection,
            "effect": round(theme.effect, 3),
            "q_value": round(theme.q_value, 4),
            "helper_mean_effect": round(theme.helper_mean_effect, 3) if theme.helper_mean_effect is not None else None,
            "leading_edges": list(theme.leading_edges),
            "top_pathways": [
                {
                    "term": record.term,
                    "collection": record.source,
                    "nes": round(record.nes, 3),
                    "q_value": round(record.q_value, 4),
                    "size": record.size,
                }
                for record in theme.top_pathways
            ],
        }


def run_pipeline(
    gsea_csv: Path,
    esea_csv: Path | None,
    background_txt: Path,
    output_dir: Path,
    llm: GeminiLLM,
    critic_llm: GeminiLLM | None = None,
    analysis_settings: AnalysisSettings | None = None,
    gsea_only: bool = False,
) -> PipelineArtifacts:
    pipeline = Pipeline(llm=llm, analysis_settings=analysis_settings, critic_llm=critic_llm)
    return pipeline.run(gsea_csv, esea_csv, background_txt, output_dir, gsea_only=gsea_only)
