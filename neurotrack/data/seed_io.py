"""Seed JSON read/write utilities for neurotrack UI workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping


SEED_JSON_VERSION = 1
COORDINATE_ORDER = "zyx"


def _normalize_seed_rows(seed_rows: Iterable[Iterable[float]]) -> List[List[float]]:
    normalized: List[List[float]] = []
    for row in seed_rows:
        values = list(row)
        if len(values) != 3:
            raise ValueError(f"Each seed must have exactly 3 values (z, y, x), got: {values}")
        normalized.append([float(values[0]), float(values[1]), float(values[2])])
    return normalized


def load_seeds_json(seeds_json_path: str | Path) -> Dict[str, List[List[float]]]:
    """Load seeds keyed by image relative path from JSON.

    Expected schema:
    {
      "version": 1,
      "coordinate_order": "zyx",
      "key_type": "relative_path",
      "seeds": {
        "path/to/image.tif": [[z, y, x], ...]
      }
    }
    """
    path = Path(seeds_json_path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, Mapping):
        raise ValueError("Seed JSON must be an object at the top level.")

    coordinate_order = str(payload.get("coordinate_order", "")).lower()
    if coordinate_order and coordinate_order != COORDINATE_ORDER:
        raise ValueError(
            f"Unsupported coordinate_order '{coordinate_order}'. Expected '{COORDINATE_ORDER}'."
        )

    key_type = str(payload.get("key_type", "relative_path")).lower()
    if key_type != "relative_path":
        raise ValueError("Seed JSON key_type must be 'relative_path'.")

    seeds_obj = payload.get("seeds", {})
    if not isinstance(seeds_obj, Mapping):
        raise ValueError("Seed JSON field 'seeds' must be an object mapping image keys to seed lists.")

    normalized: Dict[str, List[List[float]]] = {}
    for image_key, seed_rows in seeds_obj.items():
        if not isinstance(image_key, str):
            raise ValueError("Seed JSON image keys must be strings.")
        if seed_rows is None:
            normalized[image_key] = []
            continue
        if not isinstance(seed_rows, list):
            raise ValueError(f"Seed rows for '{image_key}' must be a list.")
        normalized[image_key] = _normalize_seed_rows(seed_rows)

    return normalized


def save_seeds_json(
    seeds_json_path: str | Path,
    seeds_by_relative_path: Mapping[str, Iterable[Iterable[float]]],
) -> None:
    """Write seeds keyed by image relative path to JSON in (z, y, x) order."""
    out_path = Path(seeds_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seeds_payload = {
        key: _normalize_seed_rows(rows)
        for key, rows in seeds_by_relative_path.items()
    }

    payload = {
        "version": SEED_JSON_VERSION,
        "coordinate_order": COORDINATE_ORDER,
        "key_type": "relative_path",
        "seeds": dict(sorted(seeds_payload.items())),
    }

    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
