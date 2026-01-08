"""Typed records used across the pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class GSEARecord:
    term: str
    source: str | None
    nes: float
    q_value: float
    size: int
    direction: str
    score: float
    leading_edge: Tuple[str, ...]


@dataclass(frozen=True)
class ESEARecord:
    cell_type: str
    effect_size: float
    q_value: float
    catalog: str | None
    direction: str


@dataclass(frozen=True)
class HelperRecord:
    helper_name: str
    helper_class: str
    tf_family: str | None
    nes: float
    q_value: float
    direction: str
    size: int | None
    top_hallmark: str | None


@dataclass
class Theme:
    label: str
    direction: str
    terms: List[GSEARecord]

    def top_terms(self, n: int) -> List[GSEARecord]:
        return sorted(self.terms, key=lambda rec: rec.score, reverse=True)[:n]


@dataclass
class ThemeSummary:
    theme_id: str
    label: str
    direction: str
    collection: str | None
    effect: float
    q_value: float
    top_pathways: List[GSEARecord]
    leading_edges: tuple[str, ...]
    helper_mean_effect: float | None = None


@dataclass
class HelperClaim:
    theme_id: str
    theme_label: str
    helper_name: str | None
    helper_class: str
    direction: str
    function_phrases: tuple[str, ...]
    rationale: str
    confidence: str
    helper_top_hallmark: str | None = None
    helper_effect: float | None = None
    helper_q_value: float | None = None
    claim_id: str | None = None


@dataclass
class EvidenceBundle:
    claim: HelperClaim
    evidence_snippets: tuple[str, ...]
    alternative: str
    gaps: str
    predictions: tuple[str, ...]
    raw_entry: dict | None = None


@dataclass
class ClaimEvidence:
    claim: HelperClaim
    evidence_snippets: tuple[str, ...]
    alternative: str
    gaps: str
    predictions: tuple[str, ...]
    verdict: str
    verdict_reason: str
