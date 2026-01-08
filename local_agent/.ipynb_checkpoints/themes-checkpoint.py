"""Theme grouping and ranking logic."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List

from .config import DEFAULT_THEME_CAP
from .data_models import GSEARecord, Theme

THEME_RULES: Dict[str, Iterable[str]] = {
    "TNF–NF-κB / inflammatory": ["TNF", "NFkB", "NFKB", "INFLAM", "TNFA"],
    "IFN / JAK-STAT": ["INTERFERON", "IFN", "JAK", "STAT"],
    "E2F / MYC cell cycle": ["E2F", "MYC", "S_PHASE", "CELL_CYCLE"],
    "DNA replication & repair": ["DNA_REP", "DNA REPLICATION", "REPAIR", "CHK"],
    "G2/M checkpoint": ["G2M", "MITOTIC", "CHROMOSOME", "SEGREG"],
    "Apoptosis / p53": ["P53", "APOP", "DNA_DAMAGE_RESPONSE"],
    "TNF stress & apoptosis": ["TNFA", "DEATH", "CASP"],
    "EMT / ECM / adhesion": ["EMT", "EXTRACELL", "ECM", "ADHESION", "MATRIX"],
    "Lineage / differentiation": ["ERYTH", "MYELOID", "LYMPH", "STEM", "DIFFERENT"],
    "Metabolism": ["METAB", "OXIDATIVE", "MITO", "GLYCOL"],
}


def normalize_term(term: str) -> str:
    return term.upper().replace("-", "_")


def match_theme(term: str) -> str | None:
    normalized = normalize_term(term)
    for label, keywords in THEME_RULES.items():
        for key in keywords:
            if key in normalized:
                return label
    return None


def group_by_theme(records: List[GSEARecord], direction: str) -> List[Theme]:
    grouped: Dict[str, List[GSEARecord]] = defaultdict(list)
    fallback: List[GSEARecord] = []
    for record in records:
        label = match_theme(record.term)
        if label:
            grouped[label].append(record)
        else:
            fallback.append(record)
    if not grouped and fallback:
        grouped[f"Top {direction} signals"] = fallback
    elif fallback:
        grouped.setdefault(f"Additional {direction} signals", []).extend(fallback)
    themes = [Theme(label=label, direction=direction, terms=items) for label, items in grouped.items()]
    themes.sort(key=lambda theme: sum(rec.score for rec in theme.terms) / len(theme.terms), reverse=True)
    return themes[:DEFAULT_THEME_CAP]
