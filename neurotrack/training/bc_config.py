"""Configuration model for behavior cloning and DAgger training."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


def _get_param(params: Dict[str, Any], *names: str, default=None):
    for name in names:
        if name in params:
            return params[name]
    return default


@dataclass
class BCTrainConfig:
    img_dir: str
    swc_dir: str
    out_dir: str
    name: str
    target_step_len: float = 5.0
    step_width: float = 2.0
    batch_size: int = 64
    lr: float = 0.001
    total_steps: int = 100000
    repeat_starts: bool = True
    branching: bool = False
    rng_seed: int = 1
    start_complexity: float = 0.0
    start_idx: int = 0
    crop_size: int = 128
    patches_per_image: int = 10
    save_every_updates: int = 1
    policy_weights: Optional[str] = None
    seeds_path: Optional[str] = None
    root_sampling_probability: Optional[float] = None
    soma_sample_radius: float = 0.0
    random_offset: float = 0.0
    crop_patches: bool = True
    dagger_rounds: int = 0
    steps_per_round: int = 100000
    epochs_per_update: int = 1
    steps_per_update: int = 100000
    warmstart_steps: int = 100000
    beta_schedule: str = "linear"
    beta_start: float = 1.0
    beta_end: float = 0.0
    beta_decay: float = 0.5
    beta_step: float = 0.1
    buffer_capacity: int = 100000
    continue_target_norm_threshold: Optional[float] = None
    continue_weight: float = 1.0
    norm_floor: float = 0.0
    norm_floor_weight: float = 0.0
    stop_violation_weight: float = 1.0
    objective_mode: str = "norm_floor"
    continue_direction_weight: float = 1.0
    norm_cls_weight: float = 1.0
    norm_cls_temperature: float = 0.25
    norm_margin_weight: float = 1.0
    stop_margin: float = 0.1
    continue_margin: float = 0.1
    stall_threshold: float = 1.0
    max_len: int = 1000
    max_paths: int = 1000
    gamma: float = 0.0

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "BCTrainConfig":
        """Create a config from a params dict using canonical parameter names."""
        img_dir = _get_param(params, "img_dir", "imgdir")
        swc_dir = _get_param(params, "swc_dir", "swcdir")
        out_dir = _get_param(params, "out_dir", "outdir")
        name = _get_param(params, "name")
        if img_dir is None or swc_dir is None or out_dir is None or name is None:
            raise ValueError("Config must define img_dir, swc_dir, out_dir, and name.")

        total_steps = int(_get_param(params, "total_steps", default=100000))
        steps_per_round = int(_get_param(params, "steps_per_round", default=100000))

        continue_target_norm_threshold_raw = _get_param(params, "continue_target_norm_threshold", default=None)

        config = cls(
            img_dir=str(img_dir),
            swc_dir=str(swc_dir),
            out_dir=str(out_dir),
            name=str(name),
            target_step_len=float(_get_param(params, "target_step_len", default=5.0)),
            step_width=float(_get_param(params, "step_width", default=2.0)),
            batch_size=int(_get_param(params, "batch_size", default=64)),
            lr=float(_get_param(params, "lr", "learning_rate", default=0.001)),
            total_steps=total_steps,
            repeat_starts=bool(_get_param(params, "repeat_starts", default=True)),
            branching=bool(_get_param(params, "branching", default=False)),
            rng_seed=int(_get_param(params, "rng_seed", default=1)),
            start_complexity=float(_get_param(params, "start_complexity", default=1.0)),
            start_idx=int(_get_param(params, "start_idx", default=0)),
            crop_size=int(_get_param(params, "crop_size", default=128)),
            patches_per_image=int(_get_param(params, "patches_per_image", default=10)),
            save_every_updates=int(_get_param(params, "save_every_updates", "save_every_buffer_fills", default=1)),
            policy_weights=_get_param(params, "policy_weights"),
            seeds_path=_get_param(params, "seeds_path"),
            root_sampling_probability=_get_param(params, "root_sampling_probability"),
            soma_sample_radius=float(_get_param(params, "soma_sample_radius", default=0.0)),
            random_offset=float(_get_param(params, "random_offset", default=0.0)),
            crop_patches=bool(_get_param(params, "crop_patches", default=True)),
            dagger_rounds=int(_get_param(params, "dagger_rounds", default=0)),
            steps_per_round=steps_per_round,
            epochs_per_update=int(_get_param(params, "epochs_per_update", "epochs_per_buffer_fill", default=1)),
            steps_per_update=int(_get_param(params, "steps_per_update", "save_every_steps", default=100000)),
            warmstart_steps=int(_get_param(params, "warmstart_steps", default=100000)),
            beta_schedule=str(_get_param(params, "beta_schedule", default="linear")),
            beta_start=float(_get_param(params, "beta_start", default=1.0)),
            beta_end=float(_get_param(params, "beta_end", default=0.0)),
            beta_decay=float(_get_param(params, "beta_decay", default=0.8)),
            beta_step=float(_get_param(params, "beta_step", default=0.5)),
            buffer_capacity=int(_get_param(params, "buffer_capacity", default=100000)),
            continue_target_norm_threshold=None if continue_target_norm_threshold_raw is None else float(continue_target_norm_threshold_raw),
            continue_weight=float(_get_param(params, "continue_weight", default=1.0)),
            norm_floor=float(_get_param(params, "norm_floor", default=0.0)),
            norm_floor_weight=float(_get_param(params, "norm_floor_weight", default=0.0)),
            stop_violation_weight=float(_get_param(params, "stop_violation_weight", default=1.0)),
            objective_mode=str(_get_param(params, "objective_mode", default="norm_floor")),
            continue_direction_weight=float(_get_param(params, "continue_direction_weight", default=1.0)),
            norm_cls_weight=float(_get_param(params, "norm_cls_weight", default=1.0)),
            norm_cls_temperature=float(_get_param(params, "norm_cls_temperature", default=0.25)),
            norm_margin_weight=float(_get_param(params, "norm_margin_weight", default=1.0)),
            stop_margin=float(_get_param(params, "stop_margin", default=0.1)),
            continue_margin=float(_get_param(params, "continue_margin", default=0.1)),
            stall_threshold=float(_get_param(params, "stall_threshold", default=1.0)),
            max_len=int(_get_param(params, "max_len", default=1000)),
            max_paths=int(_get_param(params, "max_paths", default=1000)),
            gamma=float(_get_param(params, "gamma", default=0.0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be > 0.")
        if self.lr <= 0:
            raise ValueError("lr must be > 0.")
        if self.crop_size <= 0:
            raise ValueError("crop_size must be > 0.")
        if self.patches_per_image <= 0:
            raise ValueError("patches_per_image must be > 0.")
        if self.save_every_updates <= 0:
            raise ValueError("save_every_updates must be > 0.")
        if self.steps_per_round <= 0:
            raise ValueError("steps_per_round must be > 0.")
        if self.epochs_per_update <= 0:
            raise ValueError("epochs_per_update must be > 0.")
        if self.steps_per_update <= 0:
            raise ValueError("steps_per_update must be > 0.")
        if self.warmstart_steps < 0:
            raise ValueError("warmstart_steps must be >= 0.")
        if self.buffer_capacity <= 0:
            raise ValueError("buffer_capacity must be > 0.")
        if self.max_len <= 0 or self.max_paths <= 0:
            raise ValueError("max_len and max_paths must be > 0.")
        if self.start_idx < 0:
            raise ValueError("start_idx must be >= 0.")
        if self.stop_violation_weight < 0:
            raise ValueError("stop_violation_weight must be >= 0.")
        if self.objective_mode not in {"norm_floor", "norm_classifier_margin", "direction_sse"}:
            raise ValueError(
                "objective_mode must be one of: {'norm_floor', 'norm_classifier_margin', 'direction_sse'}"
            )
        if self.continue_direction_weight < 0:
            raise ValueError("continue_direction_weight must be >= 0.")
        if self.norm_cls_weight < 0:
            raise ValueError("norm_cls_weight must be >= 0.")
        if self.norm_cls_temperature <= 0:
            raise ValueError("norm_cls_temperature must be > 0.")
        if self.norm_margin_weight < 0:
            raise ValueError("norm_margin_weight must be >= 0.")
        if self.stop_margin < 0:
            raise ValueError("stop_margin must be >= 0.")
        if self.continue_margin < 0:
            raise ValueError("continue_margin must be >= 0.")

    def to_log_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["BCTrainConfig"]
