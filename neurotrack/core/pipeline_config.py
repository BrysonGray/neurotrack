"""Shared pipeline configuration utilities.

Provides:
- ``flexible_image_key_lookup`` – tolerant key lookup across image-root changes.
- ``normalize_null_string``      – collapse empty / null-sentinel strings to None.
- ``PostprocessConfig``           – typed post-processing and evaluation parameters.
- ``load_pipeline_config``        – load a JSON config, apply defaults, and
                                     canonicalize key aliases.

Used by both ``inference_eval_pipeline`` and ``interactive_tracing_pipeline`` to
avoid duplicating config-loading and postprocess-parameter logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# Path-valued keys that may arrive as empty / null-sentinel strings and should
# be normalized to ``None``.
_NULL_STRING_PATH_KEYS: frozenset = frozenset({
    "swc_dir",
    "seeds_path",
    "scales_path",
    "img_dir",
    "out_dir",
    "sac_weights",
})


def normalize_null_string(value: Any) -> Optional[str]:
    """Return ``None`` when *value* is a blank or null-sentinel string.

    Handles the common JSON pattern where a user writes ``"swc_dir": ""`` or
    ``"swc_dir": "none"`` to indicate that the field is absent.
    """
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return value


def flexible_image_key_lookup(mapping: Dict, query_key: str, default=None):
    """Flexible key lookup that survives image-root changes.

    Tries (in order):
    1. Exact key match.
    2. One key is a path-suffix of the other (e.g. ``images/a.tif`` ↔ ``a.tif``).
    3. Stem match (filename without extension).

    Returns *default* when nothing matches.
    """
    if query_key in mapping:
        return mapping[query_key]

    query_parts = Path(query_key).parts
    for key, value in mapping.items():
        key_parts = Path(key).parts
        short, long = (
            (query_parts, key_parts)
            if len(query_parts) <= len(key_parts)
            else (key_parts, query_parts)
        )
        if long[-len(short):] == short:
            return value
        
    for key, value in mapping.items():
        if (Path(key).with_suffix('') == Path(query_key)
                or Path(query_key).with_suffix('') == Path(key)):
            return value

    return default


@dataclass
class PostprocessConfig:
    """Parameters controlling post-processing and evaluation steps.

    Separating these from the trace / inference params makes defaults explicit
    at the call site and prevents silent fallback to hardcoded values buried
    inside ``process_results``.

    ``scales_path`` is an optional path to a JSON file whose keys are TIFF
    filenames (or relative paths) and whose values are the x-y pixel size in
    physical units.  When provided, the distance-based parameters
    (``min_branch_length``, ``resampling_step_size``, ``smoothing_window``,
    ``overlap_distance_threshold``, ``distance_threshold``) are divided by the
    matching scale before being passed to ``process_results`` / evaluation, so
    that thresholds expressed in physical units are correctly converted to
    voxels.
    """

    # --- smoothing / merging ---
    min_branch_length: float = 5.0
    resampling_step_size: float = 4.0
    smoothing_window: int = 5
    overlap_threshold: float = 0.5
    overlap_distance_threshold: float = 1.0
    # --- evaluation ---
    distance_threshold: float = 1.0
    # --- optional per-image scale lookup ---
    scales_path: Optional[str] = None

    # Internal cache — not part of the public API; populated lazily.
    _scales_cache: Optional[Dict[str, float]] = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def from_config(cls, config: Dict[str, object]) -> "PostprocessConfig":
        """Build a ``PostprocessConfig`` from a flat config dict."""
        eval_distance_threshold = config.get("distance_threshold", None)
        if eval_distance_threshold is None:
            eval_distance_threshold = config.get("eval_distance_threshold", 1.0)
        return cls(
            min_branch_length=float(config.get("min_branch_length", 5.0)),
            resampling_step_size=float(config.get("resampling_step_size", 4.0)),
            smoothing_window=int(config.get("smoothing_window", 5)),
            overlap_threshold=float(config.get("overlap_threshold", 0.5)),
            overlap_distance_threshold=float(config.get("overlap_distance_threshold", 1.0)),
            distance_threshold=float(eval_distance_threshold),
            scales_path=config.get("scales_path", None),
        )

    def to_dict(self) -> Dict[str, object]:
        """Return the postprocess parameters as a plain dict for ``process_results``."""
        return {
            "min_branch_length": self.min_branch_length,
            "resampling_step_size": self.resampling_step_size,
            "smoothing_window": self.smoothing_window,
            "overlap_threshold": self.overlap_threshold,
            "overlap_distance_threshold": self.overlap_distance_threshold,
        }

    def _load_scales(self) -> Dict[str, float]:
        """Load and cache the scales JSON (keyed by TIFF filename / relative path)."""
        if self._scales_cache is not None:
            return self._scales_cache
        if not self.scales_path:
            self._scales_cache = {}
            return self._scales_cache
        path = Path(self.scales_path)
        if not path.exists():
            raise FileNotFoundError(f"scales_path not found: {path}")
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise ValueError(
                "scales JSON must be a flat object mapping filenames to scale values."
            )
        self._scales_cache = {k: float(v) for k, v in raw.items()}
        return self._scales_cache

    def get_scale_for_image(self, image_key: str) -> float:
        """Return the x-y pixel size for *image_key*, or 1.0 if not found / no scales_path."""
        if not self.scales_path:
            return 1.0
        scales = self._load_scales()
        scale = flexible_image_key_lookup(scales, image_key, default=None)
        if scale is None:
            import warnings
            warnings.warn(
                f"No scale entry found for image '{image_key}' in scales file "
                f"'{self.scales_path}'. Defaulting to scale=1.0 (no unit conversion).",
                stacklevel=2,
            )
            return 1.0
        print(f"[PostprocessConfig] Scale factor for '{image_key}': {float(scale):.6g} "
              f"(from '{self.scales_path}')")
        return float(scale)

    def scaled_params_for_image(self, image_key: str) -> Dict[str, object]:
        """Return ``to_dict()`` with distance-based params divided by the image scale.

        ``smoothing_window`` stays as an ``int`` (rounded after division).
        ``overlap_threshold`` is a ratio and is therefore *not* scaled.
        """
        scale = self.get_scale_for_image(image_key)
        if scale == 1.0:
            return self.to_dict()
        return {
            "min_branch_length": self.min_branch_length / scale,
            "resampling_step_size": self.resampling_step_size,
            "smoothing_window": max(1, round(self.smoothing_window)),
            "overlap_threshold": self.overlap_threshold,
            "overlap_distance_threshold": self.overlap_distance_threshold / scale,
        }


def load_pipeline_config(
    config_path: str,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """Load a JSON config file, apply *defaults*, and canonicalize key aliases.

    Steps applied (in order):

    1. Parse the JSON object.
    2. Fill in any missing keys from *defaults*.
    3. Canonicalize ``eval_distance_threshold`` → ``distance_threshold``
       (only when ``distance_threshold`` is absent from the raw JSON).
    4. Normalize null-sentinel strings to ``None`` for all known path keys.

    Parameters
    ----------
    config_path:
        Path to the JSON configuration file.
    defaults:
        Mapping of key → default value applied for keys absent in the JSON.

    Returns
    -------
    dict
        Merged and normalized configuration.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with path.open("r", encoding="utf-8") as fh:
        config: Dict[str, Any] = json.load(fh)
    if not isinstance(config, dict):
        raise ValueError("Config file must contain a JSON object.")

    # Apply defaults for absent keys.
    for key, value in defaults.items():
        if key not in config:
            config[key] = value

    # Key alias: eval_distance_threshold → distance_threshold.
    if "distance_threshold" not in config and "eval_distance_threshold" in config:
        config["distance_threshold"] = config["eval_distance_threshold"]

    # Normalize null-sentinel string paths to None.
    for key in _NULL_STRING_PATH_KEYS:
        if key in config:
            config[key] = normalize_null_string(config[key])

    return config
