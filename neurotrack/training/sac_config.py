"""Configuration model for Soft Actor-Critic training."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


def _get_param(params: Dict[str, Any], *names: str, default=None):
    for name in names:
        if name in params:
            return params[name]
    return default


@dataclass
class SACTrainConfig:
    img_dir: str
    swc_dir: str
    outdir: str
    name: str
    target_step_len: float = 1.0
    step_width: float = 1.0
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    lr: float = 0.001
    n_episodes: int = 100
    init_temperature: float = 0.005
    target_entropy: float = 0.0
    update_alpha: bool = True
    repeat_starts: bool = True
    branching: bool = False
    seeds_path: Optional[str] = None
    root_sampling_probability: Optional[float] = None
    soma_sample_radius: float = 0.0
    random_offset: float = 0.0
    rng_seed: int = 1
    start_complexity: float = 0.0
    start_idx: int = 0
    crop_patches: bool = True
    sac_weights: Optional[str] = None
    patch_radius: int = 17
    in_channels: int = 2
    crop_size: int = 128
    patches_per_image: int = 10
    max_len: int = 1000
    max_paths: int = 1000
    replay_capacity: int = 10000
    replay_alpha: float = 0.8
    update_after: int = 256
    updates_per_step: int = 1
    update_every: int = 1
    dynamic_complexity: bool = True
    show: bool = True
    pause_after_episode: bool = False
    show_live: bool = False
    pause_after_step: bool = False

    @classmethod
    def from_params(cls, params: Dict[str, Any]) -> "SACTrainConfig":
        img_dir = _get_param(params, "img_dir")
        swc_dir = _get_param(params, "swc_dir")
        outdir = _get_param(params, "outdir", "out_dir")
        name = _get_param(params, "name")
        if img_dir is None or swc_dir is None or outdir is None or name is None:
            raise ValueError("Config must define img_dir, swc_dir, outdir, and name.")

        config = cls(
            img_dir=str(img_dir),
            swc_dir=str(swc_dir),
            outdir=str(outdir),
            name=str(name),
            target_step_len=float(_get_param(params, "target_step_len", "step_size", default=1.0)),
            step_width=float(_get_param(params, "step_width", default=1.0)),
            batch_size=int(_get_param(params, "batch_size", "batchsize", default=256)),
            gamma=float(_get_param(params, "gamma", default=0.99)),
            tau=float(_get_param(params, "tau", default=0.005)),
            lr=float(_get_param(params, "lr", "learning_rate", default=0.001)),
            n_episodes=int(_get_param(params, "n_episodes", "epochs", default=100)),
            init_temperature=float(_get_param(params, "init_temperature", default=0.005)),
            target_entropy=float(_get_param(params, "target_entropy", default=0.0)),
            update_alpha=bool(_get_param(params, "update_alpha", default=True)),
            repeat_starts=bool(_get_param(params, "repeat_starts", default=True)),
            branching=bool(_get_param(params, "branching", default=False)),
            seeds_path=_get_param(params, "seeds_path"),
            root_sampling_probability=_get_param(params, "root_sampling_probability"),
            soma_sample_radius=float(_get_param(params, "soma_sample_radius", default=0.0)),
            random_offset=float(_get_param(params, "random_offset", default=0.0)),
            rng_seed=int(_get_param(params, "rng_seed", default=1)),
            start_complexity=float(_get_param(params, "start_complexity", default=0.0)),
            start_idx=int(_get_param(params, "start_idx", default=0)),
            crop_patches=bool(_get_param(params, "crop_patches", default=True)),
            sac_weights=_get_param(params, "sac_weights"),
            patch_radius=int(_get_param(params, "patch_radius", default=17)),
            in_channels=int(_get_param(params, "in_channels", default=2)),
            crop_size=int(_get_param(params, "crop_size", default=128)),
            patches_per_image=int(_get_param(params, "patches_per_image", default=10)),
            max_len=int(_get_param(params, "max_len", default=1000)),
            max_paths=int(_get_param(params, "max_paths", default=1000)),
            replay_capacity=int(_get_param(params, "replay_capacity", default=10000)),
            replay_alpha=float(_get_param(params, "replay_alpha", default=0.8)),
            update_after=int(_get_param(params, "update_after", default=256)),
            updates_per_step=int(_get_param(params, "updates_per_step", default=1)),
            update_every=int(_get_param(params, "update_every", default=1)),
            dynamic_complexity=bool(_get_param(params, "dynamic_complexity", default=True)),
            show=bool(_get_param(params, "show", default=True)),
            pause_after_episode=bool(_get_param(params, "pause_after_episode", default=False)),
            show_live=bool(_get_param(params, "show_live", default=False)),
            pause_after_step=bool(_get_param(params, "pause_after_step", default=False)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if self.lr <= 0:
            raise ValueError("lr must be > 0.")
        if self.n_episodes <= 0:
            raise ValueError("n_episodes must be > 0.")
        if self.patch_radius <= 0 or self.in_channels <= 0:
            raise ValueError("patch_radius and in_channels must be > 0.")
        if self.crop_size <= 0 or self.patches_per_image <= 0:
            raise ValueError("crop_size and patches_per_image must be > 0.")
        if self.max_len <= 0 or self.max_paths <= 0:
            raise ValueError("max_len and max_paths must be > 0.")
        if self.replay_capacity <= 0:
            raise ValueError("replay_capacity must be > 0.")
        if self.update_after <= 0 or self.updates_per_step <= 0 or self.update_every <= 0:
            raise ValueError("update_after, updates_per_step, and update_every must be > 0.")
        if self.start_idx < 0:
            raise ValueError("start_idx must be >= 0.")

    def to_log_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["SACTrainConfig"]
