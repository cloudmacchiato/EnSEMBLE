"""Audit coverage and vocabulary for theme rules across gene set libraries."""

from __future__ import annotations

import argparse
import collections
import math
import re
from pathlib import Path
from typing import Counter, Dict, Iterable, List, Sequence, Set, Tuple

from .data_models import GSEARecord
from .themes import THEME_CONFIG, match_record


GeneSet = Tuple[GSEARecord, str]

GENERIC_STOPWORDS: Set[str] = {
    "HALLMARK",
    "SIGNALING",
    "PATHWAY",
    "MODULE",
    "SET",
    "GENE",
    "UP",
    "DN",
    "GO",
    "KEGG",
    "REACTOME",
    "BIOCARTA",
    "RESPONSE",
    "PROCESS",
    "OF",
    "AND",
    "THE",
    "CELL",
    "CELLS",
    "GOBP",
    "REGULATION",
    "POSITIVE",
    "NEGATIVE",
    "TO",
    "BY",
    "IN",
    "CANONICAL",
}


def load_gmt(path: Path) -> List[GeneSet]:
    gene_sets: List[GeneSet] = []
    if not path.exists():
        raise FileNotFoundError(f"GMT file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split("\t")
            if len(parts) < 2:
                continue
            name = parts[0].strip()
            description = parts[1].strip() if parts[1] else ""
            genes = tuple(parts[2:]) if len(parts) > 2 else ()
            record = GSEARecord(
                term=name,
                source=path.stem,
                nes=0.0,
                q_value=1.0,
                size=len(genes),
                direction="UP",
                score=1.0,
                leading_edge=genes[:20],
            )
            gene_sets.append((record, description))
    return gene_sets


def tokenize(name: str) -> List[str]:
    cleaned = []
    upper = name.upper()
    buffer: List[str] = []
    for char in upper:
        if char.isalnum():
            buffer.append(char)
        else:
            if buffer:
                cleaned.append("".join(buffer))
                buffer.clear()
    if buffer:
        cleaned.append("".join(buffer))
    return cleaned


def compute_bigrams(tokens: Sequence[str]) -> List[str]:
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


TOKEN_BLACKLIST: Set[str] = {
    "TNFA",
    "TNFA_SIGNALING",
}

_TOKEN_FINDER = re.compile(r"[A-Z0-9]{3,}")


def extract_known_tokens(rule_patterns: Iterable[str]) -> Set[str]:
    tokens: Set[str] = set()
    for pattern in rule_patterns:
        for candidate in _TOKEN_FINDER.findall(pattern.upper()):
            if len(candidate) >= 3:
                tokens.add(candidate)
    return tokens


def audit_gene_sets(gmt_paths: Sequence[Path], top_k: int = 5) -> List[str]:
    messages: List[str] = []
    theme_to_names: Dict[str, Set[str]] = collections.defaultdict(set)
    theme_to_sources: Dict[str, Counter[str]] = collections.defaultdict(collections.Counter)
    all_records: List[GeneSet] = []
    for path in gmt_paths:
        all_records.extend(load_gmt(path))

    for record, description in all_records:
        matches = match_record(record)
        if not matches:
            continue
        for label, weight in matches.items():
            if weight <= 0:
                continue
            theme_to_names[label].add(record.term)
            theme_to_sources[label][record.source or "unknown"] += 1

    rule_lookup = {rule.name: rule for rule in THEME_CONFIG.rules}

    for rule in THEME_CONFIG.rules:
        name = rule.name
        matched_terms = theme_to_names.get(name, set())
        token_counter: Counter[str] = collections.Counter()
        bigram_counter: Counter[str] = collections.Counter()
        for term in matched_terms:
            tokens = tokenize(term)
            token_counter.update(tokens)
            bigram_counter.update(compute_bigrams(tokens))
        seeds = set(rule.seed_sets)
        unmatched_seeds = sorted(seed for seed in seeds if seed not in matched_terms)
        known_tokens = extract_known_tokens(
            [pattern.regex.pattern for pattern in (rule.canonical_patterns + rule.positive_patterns)]
        )
        known_tokens |= {token for token in known_tokens}
        suggestions: List[Tuple[str, int]] = []
        threshold = max(2, math.ceil(len(matched_terms) * 0.12)) if matched_terms else 0
        for token, count in token_counter.most_common():
            if token in GENERIC_STOPWORDS or token in known_tokens or token in TOKEN_BLACKLIST:
                continue
            if count < threshold:
                continue
            suggestions.append((token, count))
        header = f"Theme: {name}"
        messages.append(header)
        messages.append("  matched_sets: {} (seeds unmatched: {})".format(
            len(matched_terms), len(unmatched_seeds)
        ))
        if unmatched_seeds:
            messages.append("    missing_seeds: {}".format(", ".join(unmatched_seeds[:5])))
        if matched_terms:
            top_tokens = ", ".join(f"{token} ({count})" for token, count in token_counter.most_common(top_k)) or "-"
            top_bigrams = ", ".join(
                f"{token} ({count})" for token, count in bigram_counter.most_common(top_k) if count > 1
            ) or "-"
            messages.append(f"  top_tokens: {top_tokens}")
            messages.append(f"  top_bigrams: {top_bigrams}")
        else:
            messages.append("  top_tokens: -")
            messages.append("  top_bigrams: -")
        if suggestions:
            suggestion_str = ", ".join(f"{token} ({count})" for token, count in suggestions[:top_k])
            messages.append(f"  suggested_synonyms: {suggestion_str}")
        else:
            messages.append("  suggested_synonyms: -")
        source_counts = theme_to_sources.get(name)
        if source_counts:
            source_str = ", ".join(f"{src}:{cnt}" for src, cnt in source_counts.most_common()[:top_k])
            messages.append(f"  library_breakdown: {source_str}")
        messages.append("")
    return messages


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gmt", nargs="+", type=Path, help="Paths to GMT files to audit")
    parser.add_argument("--top", type=int, default=5, help="Number of top tokens/bigrams to report")
    args = parser.parse_args(argv)

    messages = audit_gene_sets(args.gmt, top_k=args.top)
    for line in messages:
        print(line)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
