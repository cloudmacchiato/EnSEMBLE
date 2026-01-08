"""Configuration helpers for the Gemini agent."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeminiConfig:
    """Runtime parameters for the Gemini API client."""

    model: str = "models/gemini-flash-latest"
    api_key: str | None = None
    temperature: float = 0.3
    top_p: float = 0.95
    top_k: int = 32
    max_output_tokens: int = 4096


DEFAULT_THEME_CAP: int = 3
DEFAULT_TOP_GSEA: int = 100
DEFAULT_Q_THRESHOLD: float = 0.05
DEFAULT_EFFECT_THRESHOLD: float = 0.30
DEFAULT_BACKGROUND_KEYS = (
    "Study_ID",
    "System_Model",
    "Perturbation",
    "Assay_Context",
    "Known_Biology",
    "Key_Questions",
    "Must_Link_Cell_Types",
    "Must_Link_Pathways",
    "Red_Flag_Contradictions",
)
