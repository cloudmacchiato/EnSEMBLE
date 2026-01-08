"""Deterministic preprocessing for GSEA pathways and enhancer helper tables."""

from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .config import AnalysisSettings
from .data_models import GSEARecord, HelperRecord


GSEA_COLUMN_ALIASES = {
    "compare.list": "term",
    "term": "term",
    "nes": "NES",
    "qvalue": "q_value",
    "q_value": "q_value",
    "size": "size",
    "source": "source",
    "leadingedge": "leading_edge",
    "leading_edge": "leading_edge",
}

HELPER_COLUMN_ALIASES = {
    "compare.list": "helper_name",
    "helper_name": "helper_name",
    "helper": "helper_name",
    "class": "helper_class",
    "helper_class": "helper_class",
    "catalog": "helper_class",
    "nes": "NES",
    "effect_size": "NES",
    "score": "NES",
    "qvalue": "q_value",
    "q_value": "q_value",
    "pvalue": "p_value",
    "direction": "direction",
    "size": "size",
    "top_hallmark": "top_hallmark",
    "tf_family": "tf_family",
}


NumericKeys = {"NES", "q_value", "size"}

_DEFAULT_TF_HELPER_PATH = Path(__file__).resolve().parent / "resources" / "tf_helper_names.txt"

TF_FAMILY_PREFIXES = {
    "FOX": "FOX-family",
    "GATA": "GATA-family",
    "STAT": "STAT-family",
    "IRF": "IRF-family",
    "ETS": "ETS-family",
    "EOMES": "T-box-family",
    "TBX": "T-box-family",
    "NR": "Nuclear-receptor-family",
    "PPAR": "Nuclear-receptor-family",
    "RXR": "Nuclear-receptor-family",
    "RAR": "Nuclear-receptor-family",
    "SP1": "SP/KLF-family",
    "SP2": "SP/KLF-family",
    "SP3": "SP/KLF-family",
    "SP4": "SP/KLF-family",
    "KLF": "SP/KLF-family",
    "JUN": "AP-1-family",
    "FOS": "AP-1-family",
    "ATF": "AP-1-family",
    "MAF": "AP-1-family",
    "CEBP": "C/EBP-family",
}

TF_FAMILY_KEYWORDS = [
    ("AP-1", "AP-1-family"),
    ("BACH", "BTB-bZIP-family"),
    ("SMAD", "SMAD-family"),
    ("MYC", "MYC/MAX-family"),
    ("MAX", "MYC/MAX-family"),
    ("HOX", "HOX-family"),
]


def _load_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            normalized: Dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    continue
                normalized[key] = value.strip()
            rows.append(normalized)
    return rows


def load_tf_helper_names(path: Path | None) -> Set[str]:
    if path is None or not path.exists():
        return set()
    names: Set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if fields:
                names.add(fields[0].strip())
    return {name.lower() for name in names if name}


@lru_cache(maxsize=1)
def get_default_tf_helper_names() -> Set[str]:
    if not _DEFAULT_TF_HELPER_PATH.exists():
        return set()
    with _DEFAULT_TF_HELPER_PATH.open("r", encoding="utf-8") as handle:
        return {line.strip().lower() for line in handle if line.strip()}


def _standardize_row(row: Dict[str, Any], aliases: Dict[str, str]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in row.items():
        lower = key.lower()
        target = aliases.get(lower, key)
        if target in NumericKeys and value != "":
            try:
                normalized[target] = float(value)
            except ValueError:
                normalized[target] = value
        else:
            normalized[target] = value
    return normalized


def load_gsea(path: Path) -> List[Dict[str, Any]]:
    rows = _load_csv(path)
    standardized = [_standardize_row(row, GSEA_COLUMN_ALIASES) for row in rows]
    filtered: List[Dict[str, Any]] = []
    for row in standardized:
        q_value = row.get("q_value")
        nes_value = row.get("NES")
        if q_value is None or nes_value is None:
            continue
        if isinstance(q_value, str):
            stripped = q_value.strip()
            if not stripped or stripped.lower() == "na":
                continue
            try:
                row["q_value"] = float(stripped)
            except ValueError:
                continue
        if isinstance(nes_value, str):
            stripped = nes_value.strip()
            if not stripped or stripped.lower() == "na":
                continue
            try:
                row["NES"] = float(stripped)
            except ValueError:
                continue
        if "source" not in row:
            row["source"] = None
        if "size" in row:
            row["size"] = int(float(row["size"]))
        leading = row.get("leading_edge")
        if isinstance(leading, str):
            row["leading_edge"] = [gene.strip() for gene in leading.split(",") if gene.strip()]
        elif isinstance(leading, list):
            row["leading_edge"] = [str(gene).strip() for gene in leading if str(gene).strip()]
        else:
            row["leading_edge"] = []
        filtered.append(row)
    required = {"term", "NES", "q_value", "size"}
    for row in filtered:
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"GSEA row missing columns: {missing}")
    return filtered


def _infer_tf_family(tf_name: str | None) -> Optional[str]:
    if not tf_name:
        return None
    cleaned = tf_name.replace("-", "").replace("_", "").replace("::", "").upper()
    for prefix, family in TF_FAMILY_PREFIXES.items():
        if cleaned.startswith(prefix):
            return family
    for keyword, family in TF_FAMILY_KEYWORDS:
        if keyword in cleaned:
            return family
    return None


def _normalize_helper_class(value: str | None) -> str:
    if not value:
        return "celltype"
    lowered = value.strip().lower()
    if lowered in {"tf", "tfbs", "tf_family", "transcription_factor"}:
        return "tf_family"
    if lowered in {"cell", "celltype", "cell_type", "lineage"}:
        return "celltype"
    return lowered


def load_helpers(path: Path,
                 settings: AnalysisSettings,
                 tf_helper_names: Optional[Set[str]] = None) -> List[HelperRecord]:
    rows = _load_csv(path)
    standardized = [_standardize_row(row, HELPER_COLUMN_ALIASES) for row in rows]
    if tf_helper_names is None:
        tf_lookup = get_default_tf_helper_names()
    else:
        tf_lookup = {name.lower() for name in tf_helper_names}
    helpers: List[HelperRecord] = []
    for row in standardized:
        name = (
            row.get("helper_name")
            or row.get("Compare.List")
            or row.get("Compare_List")
            or row.get("term")
        )
        if not name:
            continue
        try:
            nes = float(row.get("NES", 0.0))
        except (TypeError, ValueError):
            continue
        q_val = row.get("q_value")
        if q_val in (None, ""):
            q_val = row.get("signed_qValue")
        try:
            q_value = float(q_val)
        except (TypeError, ValueError):
            continue
        size_val = row.get("size")
        try:
            size = int(float(size_val)) if size_val not in (None, "") else None
        except (TypeError, ValueError):
            size = None
        helper_name = name.strip() if name else None
        helper_class = _normalize_helper_class(row.get("helper_class"))
        if helper_name and tf_lookup and helper_name.lower() in tf_lookup:
            helper_class = "tf_family"
        raw_direction = row.get("direction")
        if raw_direction:
            direction = raw_direction.strip().upper()
        else:
            if nes > 0:
                direction = "UP"
            elif nes < 0:
                direction = "DOWN"
            else:
                continue
        if direction not in {"UP", "DOWN"}:
            continue
        tf_family = row.get("tf_family")
        if helper_class == "tf_family":
            if not tf_family and helper_name:
                tf_family = _infer_tf_family(helper_name)
        top_hallmark = row.get("top_hallmark") or row.get("TopHallmark")
        helpers.append(
            HelperRecord(
                helper_name=helper_name or str(name),
                helper_class=helper_class,
                tf_family=tf_family,
                nes=nes,
                q_value=q_value,
                direction=direction,
                size=size,
                top_hallmark=top_hallmark if top_hallmark else None,
            )
        )
    if not helpers:
        return []
    ranked: Dict[str, List[HelperRecord]] = {"UP": [], "DOWN": []}
    for record in helpers:
        ranked.setdefault(record.direction, []).append(record)
    capped: List[HelperRecord] = []
    cap = max(0, settings.helper_max_per_direction)
    for direction in ("UP", "DOWN"):
        bucket = ranked.get(direction, [])
        bucket.sort(key=lambda rec: (rec.q_value, -abs(rec.nes)))
        if cap:
            bucket = bucket[:cap]
        capped.extend(bucket)
    return capped


def split_gsea_by_direction(rows: List[Dict[str, Any]], settings: AnalysisSettings) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {"UP": [], "DOWN": []}
    raw_buckets: Dict[str, List[Dict[str, Any]]] = {"UP": [], "DOWN": []}
    threshold = settings.gsea_q_threshold
    for row in rows:
        nes = float(row["NES"])
        q_value = float(row["q_value"])
        direction = "UP" if nes > 0 else "DOWN" if nes < 0 else "NEUTRAL"
        if direction == "NEUTRAL":
            continue
        raw_buckets.setdefault(direction, []).append(row)
        if q_value <= threshold:
            buckets.setdefault(direction, []).append(row)
    result: Dict[str, List[Dict[str, Any]]] = {}
    min_count = max(0, settings.gsea_min_per_direction)
    top_n = settings.gsea_top_n
    for direction in ("UP", "DOWN"):
        selected = list(buckets.get(direction, []))
        full = raw_buckets.get(direction, [])
        if full and len(selected) < min_count:
            fallback_pool = sorted(full, key=lambda rec: (float(rec["q_value"]), -abs(float(rec["NES"]))))
            seen = {id(rec) for rec in selected}
            for candidate in fallback_pool:
                if id(candidate) in seen:
                    continue
                selected.append(candidate)
                seen.add(id(candidate))
                if len(selected) >= min_count:
                    break
        selected.sort(key=lambda rec: abs(float(rec["NES"])), reverse=True)
        if selected:
            if top_n and top_n > 0:
                result[direction] = selected[:top_n]
            else:
                result[direction] = selected
    return result


def score_gsea_row(row: Dict[str, Any]) -> float:
    nes = float(row["NES"])
    q_value = max(float(row["q_value"]), 1e-12)
    size = max(int(row["size"]), 1)
    return abs(nes)


def to_gsea_records(rows: List[Dict[str, Any]], direction: str) -> List[GSEARecord]:
    records: List[GSEARecord] = []
    for row in rows:
        score = score_gsea_row(row)
        source = row.get("source")
        leading = row.get("leading_edge") or []
        trimmed_leading = [gene for gene in leading[:20]]
        records.append(
            GSEARecord(
                term=str(row["term"]),
                source=str(source) if source else None,
                nes=float(row["NES"]),
                q_value=float(row["q_value"]),
                size=int(row["size"]),
                direction=direction,
                score=score,
                leading_edge=tuple(trimmed_leading),
            )
        )
    return sorted(records, key=lambda rec: rec.score, reverse=True)
