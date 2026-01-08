"""Parsing and validation for background text forms."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from .config import DEFAULT_BACKGROUND_KEYS


BACKGROUND_KEY_ALIASES: Dict[str, str] = {key.lower(): key for key in DEFAULT_BACKGROUND_KEYS}
# Allow common variants
BACKGROUND_KEY_ALIASES.update({
    "contrast": "Contrast",
    "contrast_info": "Contrast",
    "contrast-direction": "Contrast",
    "contrast (case vs control)": "Contrast",
    "study id": "Study_ID",
    "system model": "System_Model",
    "assay context": "Assay_Context",
    "known biology": "Known_Biology",
    "key questions": "Key_Questions",
    "must link cell types": "Cell_Types_of_Interest_(optional)",
    "must_link_cell_types": "Cell_Types_of_Interest_(optional)",
    "cell types of interest": "Cell_Types_of_Interest_(optional)",
    "cell_types_of_interest": "Cell_Types_of_Interest_(optional)",
    "expected phenotypes or trends": "Expected_Phenotypes_or_Trends_(optional, describe expectations not mandates)",
    "expected_phenotypes_or_trends": "Expected_Phenotypes_or_Trends_(optional, describe expectations not mandates)",
    "expected phenotypes": "Expected_Phenotypes_or_Trends_(optional, describe expectations not mandates)",
    "must link pathways": "Pathway_Hypotheses_(optional)",
    "must_link_pathways": "Pathway_Hypotheses_(optional)",
    "pathway hypotheses": "Pathway_Hypotheses_(optional)",
    "pathway_hypotheses": "Pathway_Hypotheses_(optional)",
    "red flag contradictions": "Red_Flag_Contradictions",
})


@dataclass
class Background:
    raw_fields: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        segments = []
        for key in DEFAULT_BACKGROUND_KEYS:
            values = self.raw_fields.get(key, [])
            if not values:
                continue
            joined = " ".join(values)
            segments.append(f"{key}: {joined}")
        return "\n".join(segments)

    def as_dict(self) -> Dict[str, str]:
        """Return a joined-string dictionary for structured prompts."""
        return {key: " ".join(values) for key, values in self.raw_fields.items() if values}


def parse_background_txt(path: Path) -> Background:
    """Parse a simple key/value text template into structured background data."""

    if not path.exists():
        raise FileNotFoundError(path)

    fields: Dict[str, List[str]] = {}
    current_key: str | None = None
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line and not line.startswith("-"):
                key, value = line.split(":", 1)
                raw_key = key.strip()
                lookup = raw_key.lower()
                canonical = BACKGROUND_KEY_ALIASES.get(lookup)
                if canonical is None and raw_key in DEFAULT_BACKGROUND_KEYS:
                    canonical = raw_key
                if canonical is None and lookup.startswith("contrast"):
                    canonical = "Contrast"
                if canonical is not None and canonical in DEFAULT_BACKGROUND_KEYS:
                    current_key = canonical
                    remainder = value.strip()
                    if current_key not in fields:
                        fields[current_key] = []
                    if remainder:
                        fields[current_key].append(remainder)
                    continue
                # Treat lines with stray colons as body text when the key is unknown
                if current_key:
                    fields.setdefault(current_key, []).append(line)
                continue
            elif line.startswith("-") and current_key:
                fields.setdefault(current_key, []).append(line.lstrip("- "))
            elif current_key:
                fields.setdefault(current_key, []).append(line)
    if not fields.get("Contrast"):
        raise ValueError("Background template must include a Contrast entry describing numerator vs reference direction.")
    return Background(raw_fields=fields)
