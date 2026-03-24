"""
Train a Soft Actor-Critic (SAC) model for neuron tracing.
"""

import argparse
from datetime import datetime
import json
import numpy as np
import os
from pathlib import Path
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
    args_json = args.json
    with open(args_json) as f:
        params = json.load(f)
    
    img_dir = params["img_dir"]
    swc_dir = params["swc_dir"]
    outdir = params["outdir"]
    name = params["name"]
    target_step_len = params["target_step_len"] if "target_step_len" in params else 1.0
    step_width = params["step_width"] if "step_width" in params else 1.0
    batch_size = params["batchsize"] if "batchsize" in params else 256
    gamma = params["gamma"] if "gamma" in params else 0.99
    tau = params["tau"] if "tau" in params else 0.005
    lr = params["lr"] if "lr" in params else 0.001
    n_episodes = params["n_episodes"] if "n_episodes" in params else 100
    init_temperature = params["init_temperature"] if "init_temperature" in params else 0.005
    target_entropy = params["target_entropy"] if "target_entropy" in params else 0.0
    update_alpha = params["update_alpha"] if "update_alpha" in params else True
    repeat_starts = params["repeat_starts"] if "repeat_starts" in params else True
    branching = params["branching"] if "branching" in params else 0
    seeds_path = params["seeds_path"] if "seeds_path" in params else None
    root_sampling_probability = params["root_sampling_probability"] if "root_sampling_probability" in params else None
    soma_sample_radius = params["soma_sample_radius"] if "soma_sample_radius" in params else 0.0
    random_offset = params["random_offset"] if "random_offset" in params else 0.0
    rng_seed = params["rng_seed"] if "rng_seed" in params else 1
    start_complexity = params["start_complexity"] if "start_complexity" in params else 0.0
    start_idx = params["start_idx"] if "start_idx" in params else 0
    print(f"Starting training with start_idx: {start_idx}")
    patch_radius = 17
    in_channels = 2
    
    rng = np.random.default_rng(rng_seed)
    dataset = NeuronPatchDataset(
        swc_dir=swc_dir,
        img_dir=img_dir,
        crop_size=128,
        patches_per_image=10,
        alpha=start_complexity,
        step_width=step_width,
        rng=rng,
        crop_patches=True,
        inference_mode=False,
        seeds_path=seeds_path,
        root_sampling_probability=root_sampling_probability,
        soma_sample_radius=soma_sample_radius,
        random_offset=random_offset,
    )

    # Create environment
    # alpha and beta removed to test new environment reward structure
    env = NeuronTrackingEnvironment(
        dataset=dataset,
        radius=patch_radius,
        target_step_len=target_step_len,
        step_width=step_width,
        max_len=1000,
        max_paths=1000,
        gamma=gamma,
        branching=branching,
        repeat_starts=repeat_starts,
        start_idx=start_idx,
        inference_mode=False
    )
    
    input_size = 2*patch_radius+1
    actor = ConvNet(chin=in_channels, chout=4)
    actor = actor.to(device=DEVICE,dtype=dtype)

    Q1 = ConvNet(chin=in_channels+3,chout=1)
    Q1 = Q1.to(device=DEVICE,dtype=dtype)
    Q2 = ConvNet(chin=in_channels+3,chout=1)
    Q2 = Q2.to(device=DEVICE,dtype=dtype)
    Q1_target = ConvNet(chin=in_channels+3,chout=1)
    Q1_target = Q1_target.to(device=DEVICE,dtype=dtype)
    Q2_target = ConvNet(chin=in_channels+3,chout=1)
    Q2_target = Q2_target.to(device=DEVICE,dtype=dtype)

    log_alpha = torch.log(torch.tensor(init_temperature).to(DEVICE))
    log_alpha.requires_grad = True

    if "sac_weights" in params:
        sac_path = params["sac_weights"]
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
    Q1_optimizer = AdamW(Q1.parameters(), lr=lr)
    Q2_optimizer = AdamW(Q2.parameters(), lr=lr)
    actor_optimizer = AdamW(actor.parameters(), lr=lr)
    log_alpha_optimizer = Adam([log_alpha], lr=lr)

    # Load optimizer states if available
    if "sac_weights" in params:
        if "Q1_optimizer_state_dict" in state_dicts:
            Q1_optimizer.load_state_dict(state_dicts["Q1_optimizer_state_dict"])
            
        if "Q2_optimizer_state_dict" in state_dicts:
            Q2_optimizer.load_state_dict(state_dicts["Q2_optimizer_state_dict"])
            
        if "actor_optimizer_state_dict" in state_dicts:
            actor_optimizer.load_state_dict(state_dicts["actor_optimizer_state_dict"])

    memory = PrioritizedReplayBuffer(10000, obs_shape=(in_channels,input_size,input_size,input_size), action_shape=(3,), alpha=0.8)

    script_path = Path(__file__).resolve()
    logdir = script_path.parent.parent / "logs" / name
    os.makedirs(logdir, exist_ok=True)
    # save input parameters for reproducibility
    with open(logdir / f"training_params_{date_time}.json", "w") as f:
        json.dump(params, f, indent=4)
    sac.train(env, actor, Q1, Q2, Q1_target, Q2_target, log_alpha,
            actor_optimizer, Q1_optimizer, Q2_optimizer, log_alpha_optimizer,
            memory, target_entropy, batch_size, outdir, logdir,
            name, gamma, tau, update_after=256, updates_per_step=1, update_every=1, n_episodes=n_episodes,
            update_alpha=update_alpha, dynamic_complexity=True, show=True, pause_after_episode=False,
            show_live=False, pause_after_step=False)
    
    print("Done!")
    
    return


if __name__ == "__main__":
    main()