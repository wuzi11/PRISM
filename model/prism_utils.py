"""Utilities for LLM pathway prior keys and dataset loading."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from .prism_model import PROGENY_PATHWAY_NAMES


def load_env_file(path: str | Path | None) -> None:
"""Load KEY=VALUE pairs from an env file into os.environ if not already set."""
    if not path or not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


DEFAULT_LLM_ENV = str(Path(__file__).resolve().parents[1] / "config" / "llm.env")


def parse_group(group: str) -> tuple[str, str, str]:
    """Parse Sci-Plex style Group: {cell_line}_{drug}_{dose}."""
    text = str(group)
    parts = text.split("_")
    if len(parts) < 3:
        raise ValueError(f"Cannot parse Group into cell/drug/dose: {group}")
    cell_line = parts[0]
    dose = parts[-1]
    drug = "_".join(parts[1:-1])
    return cell_line, drug, dose


def drug_cell_key(cell_line: str, drug_name: str) -> str:
    return f"{cell_line}|{drug_name}"


def group_to_drug_cell_key(group: str) -> str:
    cell_line, drug, _ = parse_group(group)
    return drug_cell_key(cell_line, drug)


def directions_to_prior_vector(
    directions: Mapping[str, float | str],
    pathway_names: Sequence[str] = PROGENY_PATHWAY_NAMES,
) -> list[int]:
    """Convert LLM up/down directions to {-1, 0, 1} prior vector."""
    prior = [0] * len(pathway_names)
    name_to_idx = {name: idx for idx, name in enumerate(pathway_names)}
    for pathway, direction in directions.items():
        idx = name_to_idx.get(pathway)
        if idx is None:
            continue
        if isinstance(direction, str):
            d = direction.strip().lower()
            if d in {"up", "increase", "activate", "activated", "positive", "1"}:
                prior[idx] = 1
            elif d in {"down", "decrease", "inhibit", "inhibited", "negative", "-1"}:
                prior[idx] = -1
            else:
                prior[idx] = 0
        else:
            val = float(direction)
            if val > 0:
                prior[idx] = 1
            elif val < 0:
                prior[idx] = -1
            else:
                prior[idx] = 0
    return prior


def load_pathway_prior_store(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    pathway_names = payload.get("pathway_names", list(PROGENY_PATHWAY_NAMES))
    priors = payload.get("priors", {})
    return {
        "pathway_names": list(pathway_names),
        "priors": {str(k): list(v) for k, v in priors.items()},
    }


def build_group_prior_lookup(
    priors: Mapping[str, Sequence[int | float]],
    pathway_names: Sequence[str] = PROGENY_PATHWAY_NAMES,
) -> dict[str, np.ndarray]:
    """Map Group strings to prior vectors when priors are keyed by drug|cell."""
    lookup: dict[str, np.ndarray] = {}
    for key, vector in priors.items():
        vec = np.asarray(vector, dtype=np.float32)
        if vec.shape[0] != len(pathway_names):
            raise ValueError(
                f"Prior vector for {key} has length {vec.shape[0]}, "
                f"expected {len(pathway_names)}"
            )
        if "|" in key:
            lookup[key] = vec
        else:
            lookup[key] = vec
    return lookup


def prior_vector_for_group(
    group: str,
    priors: Mapping[str, Sequence[int | float]],
    pathway_names: Sequence[str] = PROGENY_PATHWAY_NAMES,
    default: float = 0.0,
) -> np.ndarray:
    key = group_to_drug_cell_key(group)
    if key in priors:
        return np.asarray(priors[key], dtype=np.float32)
    if str(group) in priors:
        return np.asarray(priors[str(group)], dtype=np.float32)
    return np.full(len(pathway_names), default, dtype=np.float32)


def unique_drug_cell_pairs(groups: Sequence[str]) -> list[tuple[str, str, str]]:
    seen = set()
    pairs: list[tuple[str, str, str]] = []
    for group in groups:
        cell_line, drug, _ = parse_group(group)
        key = drug_cell_key(cell_line, drug)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((cell_line, drug, key))
    return pairs


def save_pathway_prior_store(
    path: str | Path,
    priors: Mapping[str, Sequence[int]],
    pathway_names: Sequence[str] = PROGENY_PATHWAY_NAMES,
    meta: dict | None = None,
):
    payload = {
        "pathway_names": list(pathway_names),
        "priors": {str(k): [int(v) for v in values] for k, values in priors.items()},
    }
    if meta:
        payload["meta"] = meta
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def tensorize_pathway_priors(
    groups: Sequence[str],
    priors: Mapping[str, Sequence[int | float]],
    pathway_names: Sequence[str] = PROGENY_PATHWAY_NAMES,
    default: float = 0.0,
) -> torch.Tensor:
    rows = [
        prior_vector_for_group(group, priors, pathway_names, default=default)
        for group in groups
    ]
    return torch.tensor(np.stack(rows, axis=0), dtype=torch.float32)
