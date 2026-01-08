"""Verification logic mirroring GeneAgent's supported/partial/refuted judgments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .config import AnalysisSettings
from .data_models import ESEARecord, Theme


@dataclass
class Claim:
    claim: str
    theme: str
    direction: str
    cell_type: str


@dataclass
class VerificationResult:
    claim: Claim
    decision: str
    reason: str
    matched_esea: Optional[ESEARecord]
    matched_terms: List[str]
    background_flag: Optional[str] = None

    def to_row(self) -> Dict[str, str]:
        se = self.matched_esea
        effect = f"{se.effect_size:+.2f}" if se else "NA"
        q_value = f"{se.q_value:.3g}" if se else "NA"
        return {
            "theme": self.claim.theme,
            "direction": self.claim.direction,
            "cell_type": self.claim.cell_type,
            "decision": self.decision,
            "effect_size": effect,
            "q_value": q_value,
            "reason": self.reason,
        }


def normalize(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = cleaned.replace("—", "-").replace("–", "-")
    cleaned = " ".join(cleaned.split())
    return cleaned


def index_esea(records: Iterable[ESEARecord]) -> Dict[str, ESEARecord]:
    return {normalize(record.cell_type): record for record in records}


def index_themes(themes: Iterable[Theme]) -> Dict[str, Theme]:
    return {normalize(theme.label): theme for theme in themes}


def evaluate_direction(effect_size: float, direction: str) -> str:
    if direction.upper() == "UP":
        return "positive" if effect_size > 0 else "opposite"
    if direction.upper() == "DOWN":
        return "negative" if effect_size < 0 else "opposite"
    return "unknown"


def decide_support(esea: Optional[ESEARecord], direction: str, settings: AnalysisSettings) -> tuple[str, str]:
    if esea is None:
        return "Refuted", "Cell type absent from ESEA table."
    direction_eval = evaluate_direction(esea.effect_size, direction)
    support_q = settings.esea_q_threshold
    support_effect = settings.esea_effect_threshold
    partial_q = settings.esea_partial_q_threshold
    partial_effect = settings.esea_partial_effect_threshold

    if direction_eval == "opposite" and esea.q_value <= support_q:
        return "Refuted", f"Effect sign ({esea.effect_size:+.2f}) opposes {direction} with q={esea.q_value:.3g}."
    if direction_eval in {"positive", "negative"}:
        if esea.q_value <= support_q and abs(esea.effect_size) >= support_effect:
            return "Supported", f"Effect {esea.effect_size:+.2f}, q={esea.q_value:.3g} meets thresholds."
        if esea.q_value <= partial_q or abs(esea.effect_size) >= partial_effect:
            return "Partial", f"Effect {esea.effect_size:+.2f}, q={esea.q_value:.3g} is borderline; treat cautiously."
        return "Refuted", f"Effect {esea.effect_size:+.2f}, q={esea.q_value:.3g} is weak for {direction}."
    return "Refuted", f"Unable to assess direction for claimed {direction}."


def downgrade(decision: str) -> str:
    if decision == "Supported":
        return "Partial"
    if decision == "Partial":
        return "Refuted"
    return decision


def apply_background_flag(decision: str, reason: str, background_flag: Optional[str]) -> tuple[str, str, Optional[str]]:
    if background_flag and background_flag.lower().startswith("yes"):
        downgraded = downgrade(decision)
        justification = background_flag.split("—", 1)[-1].strip() if "—" in background_flag else background_flag
        updated_reason = f"{reason} Background note: {justification}."
        return downgraded, updated_reason, background_flag
    return decision, reason, background_flag
