"""Configuration helpers for the Gemini agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeminiConfig:
    """Runtime parameters for the Gemini API client."""

    model: str = "models/gemini-2.5-flash"
    api_key: str | None = None
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 32
    max_output_tokens: int = 10000


@dataclass(frozen=True)
class AnalysisSettings:
    """Thresholds and caps controlling deterministic filtering."""

    gsea_top_n: int = 0
    gsea_q_threshold: float = 0.05
    gsea_min_per_direction: int = 1
    helper_q_threshold: float = 0.05
    helper_nes_threshold: float = 0.25
    helper_max_per_direction: int = 50
    helper_claims_per_theme: int = 0
    theme_cap: int = 10
    theme_top_pathways: int = 3
    theme_leading_edge_target: int = 12
    theme_cap_total: int = 20
    head_linker_theme_cap: int = 10
    head_linker_theme_cap_total: int = 20


DEFAULT_BACKGROUND_KEYS = (
    "Study_ID",
    "System_Model",
    "Perturbation",
    "Contrast",
    "Assay_Context",
    "Known_Biology",
    "Key_Questions",
    "Cell_Types_of_Interest_(optional)",
    "Expected_Phenotypes_or_Trends_(optional, describe expectations not mandates)",
    "Pathway_Hypotheses_(optional)",
    "Red_Flag_Contradictions",
)
