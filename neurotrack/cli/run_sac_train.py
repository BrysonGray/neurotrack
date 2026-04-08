"""
Train a Soft Actor-Critic (SAC) model for neuron tracing.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import numpy as np
import os
from pathlib import Path
from typing import Dict, Optional
import torch
from torch.optim.adamw import AdamW
from torch.optim.adam import Adam

from neurotrack.data import NeuronPatchDataset
from neurotrack.environments import NeuronTrackingEnvironment
from neurotrack.training import PrioritizedReplayBuffer
from neurotrack.models import ConvNet
from neurotrack.training import sac

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
date_time = datetime.now().strftime("'%Y-%m-%d_%H-%M-%S'")


def _get_param(params: dict, *names: str, default=None):
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
    def from_params(cls, params: Dict) -> "SACTrainConfig":
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

    def to_log_dict(self) -> Dict:
        return asdict(self)


def main():

    """
    Main function to train a Soft Actor-Critic (SAC) model for tractography.
    This function parses input parameters from a JSON file, initializes the environment,
    neural network models, optimizers, and other necessary components, and then trains
    the SAC model using the specified parameters.
    
    JSON Configuration Parameters
    -----------------------------
    data_dir : str
        Path to the input data directory.
    outdir : str
        Directory to save output results.
    name : str
        Name for the training session.
    step_size : float, optional
        Step size for the environment (default is 1.0).
    step_width : float, optional
        Step width for the environment (default is 1.0).
    batchsize : int, optional
        Batch size for training (default is 256).
    tau : float, optional
        Soft update parameter for target networks (default is 0.005).
    gamma : float, optional
        Discount factor for future rewards (default is 0.99).
    lr : float, optional
        Learning rate for optimizers (default is 0.001).
    alpha : float, optional
        The weight applied to the accuracy component of reward. (default is 1.0).
    beta : float, optional
        The weight applied to the reward prior (default is 1e-3).
    friction : float, optional
        Weight applied to the friction component of reward (default is 1e-4).
    n_episodes : int, optional
        Number of training episodes (default is 100).
    init_temperature : float, optional
        Initial temperature for SAC entropy (default is 0.005).
    target_entropy : float, optional
        Target entropy for SAC (default is 0.0).
    classifier_weights : str, optional
        Path to pre-trained classifier weights.
    sac_weights : str, optional
        Path to pre-trained SAC model weights.
    """
    

    parser = argparse.ArgumentParser(description="Train SAC model from a JSON config.")
    parser.add_argument('-i', '--json', type=str, required=True, help='Path to input parameters json file.')
    args = parser.parse_args()
    config_path = Path(args.json).resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    config = SACTrainConfig.from_params(params)
    print(f"Starting training with start_idx: {config.start_idx}")
    
    rng = np.random.default_rng(config.rng_seed)
    dataset = NeuronPatchDataset(
        swc_dir=config.swc_dir,
        img_dir=config.img_dir,
        crop_size=config.crop_size,
        patches_per_image=config.patches_per_image,
        alpha=config.start_complexity,
        step_width=config.step_width,
        rng=rng,
        crop_patches=config.crop_patches,
        inference_mode=False,
        seeds_path=config.seeds_path,
        root_sampling_probability=config.root_sampling_probability,
        soma_sample_radius=config.soma_sample_radius,
        random_offset=config.random_offset,
    )

    # Create environment
    # alpha and beta removed to test new environment reward structure
    env = NeuronTrackingEnvironment(
        dataset=dataset,
        radius=config.patch_radius,
        target_step_len=config.target_step_len,
        step_width=config.step_width,
        max_len=config.max_len,
        max_paths=config.max_paths,
        gamma=config.gamma,
        branching=config.branching,
        repeat_starts=config.repeat_starts,
        start_idx=config.start_idx,
        inference_mode=False
    )
    
    input_size = 2 * config.patch_radius + 1
    actor = ConvNet(chin=config.in_channels, chout=4, rng_seed=config.rng_seed)
    actor = actor.to(device=DEVICE,dtype=dtype)

    Q1 = ConvNet(chin=config.in_channels + 3, chout=1, rng_seed=config.rng_seed)
    Q1 = Q1.to(device=DEVICE,dtype=dtype)
    Q2 = ConvNet(chin=config.in_channels + 3, chout=1, rng_seed=config.rng_seed)
    Q2 = Q2.to(device=DEVICE,dtype=dtype)
    Q1_target = ConvNet(chin=config.in_channels + 3, chout=1, rng_seed=config.rng_seed)
    Q1_target = Q1_target.to(device=DEVICE,dtype=dtype)
    Q2_target = ConvNet(chin=config.in_channels + 3, chout=1, rng_seed=config.rng_seed)
    Q2_target = Q2_target.to(device=DEVICE,dtype=dtype)

    log_alpha = torch.log(torch.tensor(config.init_temperature).to(DEVICE))
    log_alpha.requires_grad = True

    if config.sac_weights is not None:
        sac_path = config.sac_weights
        state_dicts = torch.load(sac_path)#, weights_only=True)
        actor.load_state_dict(state_dicts["policy_state_dict"])
        Q1.load_state_dict(state_dicts["Q1_state_dict"])
        Q2.load_state_dict(state_dicts["Q2_state_dict"])
        
        # Load target networks if available
        if "Q1_target_state_dict" in state_dicts:
            Q1_target.load_state_dict(state_dicts["Q1_target_state_dict"])
        else:
            Q1_target.load_state_dict(Q1.state_dict())
            
        if "Q2_target_state_dict" in state_dicts:
            Q2_target.load_state_dict(state_dicts["Q2_target_state_dict"])
        else:
            Q2_target.load_state_dict(Q2.state_dict())
            
    else:
        Q1_target.load_state_dict(Q1.state_dict())
        Q2_target.load_state_dict(Q2.state_dict())

    # Initialize optimizers
    Q1_optimizer = AdamW(Q1.parameters(), lr=config.lr)
    Q2_optimizer = AdamW(Q2.parameters(), lr=config.lr)
    actor_optimizer = AdamW(actor.parameters(), lr=config.lr)
    log_alpha_optimizer = Adam([log_alpha], lr=config.lr)

    # Load optimizer states if available
    if config.sac_weights is not None:
        if "Q1_optimizer_state_dict" in state_dicts:
            Q1_optimizer.load_state_dict(state_dicts["Q1_optimizer_state_dict"])
            
        if "Q2_optimizer_state_dict" in state_dicts:
            Q2_optimizer.load_state_dict(state_dicts["Q2_optimizer_state_dict"])
            
        if "actor_optimizer_state_dict" in state_dicts:
            actor_optimizer.load_state_dict(state_dicts["actor_optimizer_state_dict"])

    memory = PrioritizedReplayBuffer(
        config.replay_capacity,
        obs_shape=(config.in_channels, input_size, input_size, input_size),
        action_shape=(3,),
        alpha=config.replay_alpha,
    )

    script_path = Path(__file__).resolve()
    logdir = script_path.parent.parent / "logs" / config.name
    os.makedirs(logdir, exist_ok=True)
    # save input parameters for reproducibility
    params_to_save = config.to_log_dict()
    params_to_save["resolved_from_config"] = str(config_path)
    with open(logdir / f"training_params_{date_time}.json", "w", encoding="utf-8") as f:
        json.dump(params_to_save, f, indent=4)
    sac.train(env, actor, Q1, Q2, Q1_target, Q2_target, log_alpha,
            actor_optimizer, Q1_optimizer, Q2_optimizer, log_alpha_optimizer,
            memory, config.target_entropy, config.batch_size, config.outdir, logdir,
            config.name, config.gamma, config.tau,
            update_after=config.update_after,
            updates_per_step=config.updates_per_step,
            update_every=config.update_every,
            n_episodes=config.n_episodes,
            update_alpha=config.update_alpha,
            dynamic_complexity=config.dynamic_complexity,
            show=config.show,
            pause_after_episode=config.pause_after_episode,
            show_live=config.show_live,
            pause_after_step=config.pause_after_step)
    
    print("Done!")
    
    return


if __name__ == "__main__":
    main()