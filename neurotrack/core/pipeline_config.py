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


def _compute_enable_flag_for_step(config: Dict[str, object], raw_config: Dict[str, object], step_name: str) -> bool:
    """Compute the default enable flag for a postprocessing step based on parameter presence.

    If the enable flag is explicitly set in the raw (pre-default) config, use that value.
    Otherwise, return True if any of the step's parameters are present in the raw config, False otherwise.
    """
    enable_key = None
    parameter_keys = []

    if step_name == "filter_branches_by_length":
        enable_key = "filter_branches_by_length"
        parameter_keys = ["min_branch_length", "max_branch_length"]
    elif step_name == "resample":
        enable_key = "resample"
        parameter_keys = ["resampling_step_size"]
    elif step_name == "smooth_paths":
        enable_key = "smooth_paths"
        parameter_keys = ["smoothing_window"]
    elif step_name == "merge_paths":
        enable_key = "enable_merge"
        parameter_keys = [
            "merge_threshold",
            "confidence_threshold",
            "mask_smoothing_size",
            "merge_timeout_seconds",
        ]
    else:
        return False  # Unknown step; default to disabled

    # If the enable flag is explicitly in the raw config, use it.
    if enable_key in raw_config:
        return bool(raw_config[enable_key])

    # Otherwise, enable if any of the parameters are present in the raw config.
    return any(key in raw_config for key in parameter_keys)


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
    ``merge_threshold``, ``distance_threshold``) are divided by the
    matching scale before being passed to ``process_results`` / evaluation, so
    that thresholds expressed in physical units are correctly converted to
    voxels.
    """

    # --- per-step enable flags (computed on-the-fly if not in config) ---
    filter_branches_by_length: bool = False
    resample: bool = False
    smooth_paths: bool = False
    merge_paths: bool = False
    # --- branch length filtering ---
    min_branch_length: float = 5.0
    max_branch_length: float = float("inf")
    # --- resampling ---
    resampling_step_size: float = 4.0
    # --- smoothing / merging ---
    enable_length_filter: bool = True
    enable_resample: bool = True
    smoothing_window: int = 5
    enable_smooth_paths: bool = True
    enable_merge: bool = True
    merge_threshold: float = 1.0
    confidence_threshold: int = 0  # min input paths supporting a node; <=1 disables
    mask_smoothing_size: int = 0  # binary close/open of the merge overlap mask; <=1 disables
    merge_timeout_seconds: float = 30.0  # <=0 disables timeout guard
    # --- evaluation ---
    distance_threshold: float = 1.0
    # --- optional per-image scale lookup ---
    scales_path: Optional[str] = None

    # Internal cache — not part of the public API; populated lazily.
    _scales_cache: Optional[Dict[str, float]] = field(
        default=None, init=False, repr=False, compare=False
    )

    @classmethod
    def from_config(cls, config: Dict[str, object], raw_config: Optional[Dict[str, object]] = None) -> "PostprocessConfig":
        """Build a ``PostprocessConfig`` from a flat config dict.

        If raw_config is provided (the original JSON before defaults), it is used
        to compute default values for per-step enable flags. If a step's enable flag
        is explicitly set, it is used; otherwise, the step is enabled if any of its
        parameters are present in the raw config.
        """
        if raw_config is None:
            raw_config = config.get("_raw_config", config)

        eval_distance_threshold = config.get("distance_threshold", None)
        if eval_distance_threshold is None:
            eval_distance_threshold = config.get("eval_distance_threshold", 1.0)

        filter_branches_by_length = _compute_enable_flag_for_step(
            config, raw_config, "filter_branches_by_length"
        )
        resample = _compute_enable_flag_for_step(config, raw_config, "resample")
        smooth_paths = _compute_enable_flag_for_step(config, raw_config, "smooth_paths")
        merge_paths = _compute_enable_flag_for_step(
            config, raw_config, "merge_paths"
        )

        return cls(
            filter_branches_by_length=filter_branches_by_length,
            resample=resample,
            smooth_paths=smooth_paths,
            merge_paths=merge_paths,
            min_branch_length=float(config.get("min_branch_length", 5.0)),
            max_branch_length=float(config.get("max_branch_length", float("inf"))),
            resampling_step_size=float(config.get("resampling_step_size", 4.0)),
            enable_length_filter=bool(config.get("enable_length_filter", filter_branches_by_length)),
            enable_resample=bool(config.get("enable_resample", resample)),
            smoothing_window=int(config.get("smoothing_window", 5)),
            enable_smooth_paths=bool(config.get("enable_smooth_paths", smooth_paths)),
            enable_merge=bool(config.get("enable_merge", merge_paths)),
            merge_threshold=float(config.get("merge_threshold", 1.0)),
            confidence_threshold=int(config.get("confidence_threshold", 0)),
            mask_smoothing_size=int(config.get("mask_smoothing_size", 0)),
            merge_timeout_seconds=float(config.get("merge_timeout_seconds", 30.0)),
            distance_threshold=float(eval_distance_threshold),
            scales_path=config.get("scales_path", None),
        )

    def to_dict(self) -> Dict[str, object]:
        """Return the postprocess parameters as a plain dict for ``process_results``."""
        return {
            "enable_length_filter": self.enable_length_filter,
            "min_branch_length": self.min_branch_length,
            "max_branch_length": self.max_branch_length,
            "enable_resample": self.enable_resample,
            "resampling_step_size": self.resampling_step_size,
            "enable_smooth_paths": self.enable_smooth_paths,
            "smoothing_window": self.smoothing_window,
            "enable_merge": self.enable_merge,
            "merge_threshold": self.merge_threshold,
            "confidence_threshold": self.confidence_threshold,
            "mask_smoothing_size": self.mask_smoothing_size,
            "merge_timeout_seconds": self.merge_timeout_seconds,
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
        """
        scale = self.get_scale_for_image(image_key)
        if scale == 1.0:
            return self.to_dict()
        scaled_max_branch_length = self.max_branch_length
        if scaled_max_branch_length != float("inf"):
            scaled_max_branch_length = self.max_branch_length / scale
        return {
            "enable_length_filter": self.enable_length_filter,
            "min_branch_length": self.min_branch_length / scale,
            "max_branch_length": scaled_max_branch_length,
            "enable_resample": self.enable_resample,
            "resampling_step_size": self.resampling_step_size,
            "enable_smooth_paths": self.enable_smooth_paths,
            "smoothing_window": max(1, round(self.smoothing_window)),
            "enable_merge": self.enable_merge,
            "merge_threshold": self.merge_threshold / scale,
            "confidence_threshold": self.confidence_threshold,
            "mask_smoothing_size": self.mask_smoothing_size,
            "merge_timeout_seconds": self.merge_timeout_seconds,
        }


def load_pipeline_config(
    config_path: str,
) -> Dict[str, Any]:
    """Load a JSON config file, apply defaults, and canonicalize key aliases.

    Steps applied (in order):

    1. Parse the JSON object.
    2. Fill in missing keys with built-in defaults.
    3. Canonicalize ``eval_distance_threshold`` → ``distance_threshold``
       (only when ``distance_threshold`` is absent from the raw JSON).
    4. Normalize null-sentinel strings to ``None`` for all known path keys.

    Parameters
    ----------
    config_path:
        Path to the JSON configuration file.
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

    # Save the raw config before applying defaults (needed for per-step enable flag computation).
    raw_config = dict(config)

    # Apply pipeline defaults for absent keys.
    config.setdefault("step_width", 2.0)
    config.setdefault("repeat_starts", False)
    config.setdefault("rng_seed", 1)
    config.setdefault("n_trials", 1)
    config.setdefault("seeds_path", None)
    config.setdefault("soma_sample_radius", 0.0)
    config.setdefault("random_offset", 0.0)
    config.setdefault("review_before_next", False)
    config.setdefault("sync", False)
    config.setdefault("run_evaluation", None)
    config.setdefault("min_branch_length", 5.0)
    config.setdefault("resampling_step_size", 4.0)
    config.setdefault("smoothing_window", 5)
    config.setdefault("merge_threshold", 5.0)
    config.setdefault("eval_distance_threshold", None)
    config.setdefault("distance_threshold", 5.0)
    config.setdefault("scales_path", None)
    config.setdefault("swc_dir", None)

    # Key alias: eval_distance_threshold → distance_threshold.
    if "distance_threshold" not in config and "eval_distance_threshold" in config:
        config["distance_threshold"] = config["eval_distance_threshold"]

    # Normalize null-sentinel string paths to None.
    for key in _NULL_STRING_PATH_KEYS:
        if key in config:
            config[key] = normalize_null_string(config[key])

    # Store raw config in a special key so PostprocessConfig can access it.
    config["_raw_config"] = raw_config

    return config
