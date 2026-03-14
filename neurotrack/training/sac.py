
"""
This module implements a Soft Actor-Critic (SAC) algorithm for training a reinforcement learning agent
to perform tractography. The main components include functions for sampling from the actor's output,
updating the Q-networks and actor, performing target network updates, and training the agent.

Version 2 (v2) includes changes to actor step to use the reward gradient.
"""

import csv
from collections import deque
from datetime import datetime
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import re
import sys
import torch
from tqdm import tqdm

from neurotrack.training.memory import ReplayBuffer, PrioritizedReplayBuffer
from neurotrack.training.env_inspector import show_state
from neurotrack.training.gif import trace_gif
from neurotrack.visualization.ortho_viewer import show_inference_overlay_and_wait
from neurotrack.environments.tracking_reward import distance_reward


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
date = datetime.now().strftime("%m-%d-%y")


def sample_from_output(out):    
    """
    A function to differentiably sample from the output tensor.
    
    Parameters
    ----------
    out : torch.Tensor
        The input tensor containing the mean and log variance components.
        The first three columns represent the mean, and the remaining columns
        represent the log variance.
    random : bool, optional
        If True, samples randomly from the distribution. Default is False.
        
    Returns
    -------
    torch.distributions.MultivariateNormal
        A multivariate normal distribution parameterized by the processed mean
        and log variance.
    """

    mean = out[:,:3] # component 0, 1 and 2
    logvar = out[:,3:] # logvar component 3

    meannorm = torch.linalg.norm(mean, dim=-1, keepdim=True)
    meannorm_ = torch.tanh(meannorm)*10 # maximum of 10
    mean = mean * meannorm_/(meannorm + torch.finfo(out.dtype).eps)
    logvar = torch.tanh(logvar)*3 - 1 # logvar between -4 and 2
    direction_dist = torch.distributions.MultivariateNormal(mean[:,:3], torch.exp(logvar)[:,None]*torch.eye(3, device=out.device)[None])

    return direction_dist


def _concat_obs_with_action(obs, action):
    """Append action components as constant channels to the observation volume."""
    action_channels = action[:, :, None, None, None].expand(-1, -1, obs.shape[2], obs.shape[3], obs.shape[4])
    return torch.cat((obs, action_channels), dim=1)


def update_Q(actor, Q1, Q1_target, Q2, Q2_target,
             obs, actions, next_target_vecs, next_target_masks, next_obs, dones,
             Q1_optimizer, Q2_optimizer, gamma,
             log_alpha, weights=None):
    """
    Perform one step of the optimization on the Q networks.
    
    Parameters
    ----------
    actor : torch.nn.Module
        The actor network used to sample next actions.
    Q1 : torch.nn.Module
        The first Q network.
    Q1_target : torch.nn.Module
        The target network for the first Q network.
    Q2 : torch.nn.Module
        The second Q network.
    Q2_target : torch.nn.Module
        The target network for the second Q network.
    obs : torch.Tensor
        The current observations.
    actions : torch.Tensor
        The actions taken.
    next_target_vecs : torch.Tensor
        The target vector candidates at the next states.
    next_target_masks : torch.Tensor
        Boolean masks selecting valid next-state target vector candidates.
    next_obs : torch.Tensor
        The next observations.
    dones : torch.Tensor
        The done flags indicating episode termination.
    Q1_optimizer : torch.optim.Optimizer
        The optimizer for the first Q network.
    Q2_optimizer : torch.optim.Optimizer
        The optimizer for the second Q network.
    gamma : float
        The discount factor.
    log_alpha : torch.Tensor
        The logarithm of the temperature parameter.
    weights : torch.Tensor, optional
        The weights for the loss function, by default None.
        
    Returns
    -------
    torch.Tensor
        The temporal difference error.
    """

    # compute targets
    with torch.no_grad():
        # sample next actions from the current policy
        actor_out = actor(next_obs)
        direction_dist = sample_from_output(actor_out)
        next_directions = direction_dist.rsample()
        logprobs = direction_dist.log_prob(next_directions)
        # get target q-values
        next_states = _concat_obs_with_action(next_obs, next_directions)

        next_rewards = distance_reward(
            next_directions,
            next_target_vecs,
            terminated=dones,
            gamma=gamma,
            valid_mask=next_target_masks,
        )
        Q1_target_vals = Q1_target(next_states) - next_rewards.unsqueeze(-1) # vector of q-values for each choice
        Q2_target_vals = Q2_target(next_states) - next_rewards.unsqueeze(-1)
        targets = gamma * torch.logical_not(dones) * (torch.minimum(Q1_target_vals, Q2_target_vals) - log_alpha.exp() * logprobs[:,None])
        
        # Check for NaN values in intermediate computations
        if torch.isnan(logprobs).any():
            print("WARNING: NaN detected in logprobs!", flush=True)
        if torch.isnan(Q1_target_vals).any():
            print("WARNING: NaN detected in Q1_target_vals!", flush=True)
        if torch.isnan(Q2_target_vals).any():
            print("WARNING: NaN detected in Q2_target_vals!", flush=True)
        if torch.isnan(log_alpha.exp()).any():
            print(f"WARNING: NaN detected in log_alpha.exp()! log_alpha={log_alpha}", flush=True)
        if torch.isnan(targets).any():
            print("WARNING: NaN detected in targets!", flush=True)
    # compute q-values to compare against targets
    current_state = _concat_obs_with_action(obs, actions)
    
    if weights is None:
        weights = torch.ones_like(targets)
    elif weights.device != targets.device:
        weights = weights.to(device=targets.device)

    Q1_vals = Q1(current_state)
    Q1_td_error = torch.abs(Q1_vals - targets).detach()
    Q1_loss = torch.mean((Q1_vals - targets)**2 * weights)
    Q1_optimizer.zero_grad()
    Q1_loss.backward()
    Q1_optimizer.step()

    Q2_vals = Q2(current_state)
    Q2_td_error = torch.abs(Q2_vals - targets).detach()
    Q2_loss = torch.mean((Q2_vals - targets)**2 * weights)
    Q2_optimizer.zero_grad()
    Q2_loss.backward()
    Q2_optimizer.step()

    td_error = torch.maximum(Q1_td_error, Q2_td_error).squeeze()
    
    # Check for NaN values in td_error
    if torch.isnan(td_error).any():
        print("WARNING: NaN detected in td_error!", flush=True)
        print(f"Q1_vals has NaN: {torch.isnan(Q1_vals).any()}", flush=True)
        print(f"Q2_vals has NaN: {torch.isnan(Q2_vals).any()}", flush=True)
        print(f"targets has NaN: {torch.isnan(targets).any()}", flush=True)
        print(f"Q1_td_error has NaN: {torch.isnan(Q1_td_error).any()}", flush=True)
        print(f"Q2_td_error has NaN: {torch.isnan(Q2_td_error).any()}", flush=True)
    
    return td_error


def update_actor(obs, target_vecs, target_masks, dones, gamma, actor,
                 actor_optimizer, Q1, Q2, log_alpha,
                 log_alpha_optimizer, target_entropy, update_alpha=True):
    """
    Update the actor network and the temperature parameter in the Soft Actor-Critic (SAC) algorithm.

    Parameters
    ----------
    obs : torch.Tensor
        The observations from the environment.
    target_vecs : torch.Tensor
        The target vector candidates from the environment.
    target_masks : torch.Tensor
        Boolean masks selecting valid target vector candidates.
    actor : torch.nn.Module
        The actor network.
    actor_optimizer : torch.optim.Optimizer
        The optimizer for the actor network.
    Q1 : torch.nn.Module
        The first Q-value network.
    Q2 : torch.nn.Module
        The second Q-value network.
    log_alpha : torch.Tensor
        The logarithm of the temperature parameter.
    log_alpha_optimizer : torch.optim.Optimizer
        The optimizer for the temperature parameter.
    target_entropy : float
        The target entropy value.

    Returns
    -------
    float
        The loss value for the actor network.
    """

    actor_out = actor(obs)
    direction_dist = sample_from_output(actor_out)
    directions = direction_dist.rsample()
    logprobs = direction_dist.log_prob(directions)
    # compute rewards
    rewards = distance_reward(
        directions,
        target_vecs,
        terminated=dones,
        gamma=gamma,
        valid_mask=target_masks,
    ).unsqueeze(-1)
    if gamma > 0:
        # get expected Q-vals
        current_state = _concat_obs_with_action(obs, directions)
        Q1_vals = Q1(current_state)
        Q2_vals = Q2(current_state)
        # entropy regularized Q values
        loss = -torch.mean(torch.minimum(Q1_vals, Q2_vals) + rewards - log_alpha.exp().detach() * logprobs[:,None]) # The loss function is multiplied by -1 to do gradient ascent instead of decent.
    else:
        loss = -torch.mean(rewards - log_alpha.exp().detach() * logprobs[:,None])
    actor_optimizer.zero_grad()
    loss.backward()
    actor_optimizer.step()

    # update temperature parameter
    if update_alpha:
        log_alpha_optimizer.zero_grad()
        entropy_term = (-logprobs - target_entropy).detach()
        
        # Check for NaN in entropy term
        if torch.isnan(entropy_term).any():
            print("WARNING: NaN detected in entropy term for alpha loss!", flush=True)
            print(f"logprobs has NaN: {torch.isnan(logprobs).any()}", flush=True)
            print(f"target_entropy: {target_entropy}", flush=True)
        
        alpha_loss = log_alpha * entropy_term.mean()
        
        # Check for NaN in alpha_loss
        if torch.isnan(alpha_loss).any():
            print("WARNING: NaN detected in alpha_loss!", flush=True)
            print(f"log_alpha: {log_alpha}", flush=True)
            print(f"entropy_term.mean(): {entropy_term.mean()}", flush=True)
        
        alpha_loss.backward()
        log_alpha_optimizer.step()

    return loss.item()


def target_update(Q1, Q2, Q1_target, Q2_target, tau):
    """
    Update the target Q-networks using Polyak averaging.

    Parameters
    ----------
    Q1 : torch.nn.Module
        The first Q network.
    Q2 : torch.nn.Module
        The second Q network.
    Q1_target : torch.nn.Module
        The target network for the first Q network.
    Q2_target : torch.nn.Module
        The target network for the second Q network.
    tau : float
        The interpolation parameter for Polyak averaging.
    """
    for Q,Q_target in zip([Q1, Q2], [Q1_target, Q2_target]):
        Q_state_dict = Q.state_dict()
        Q_target_state_dict = Q_target.state_dict()
        for key in Q_state_dict:
            Q_target_state_dict[key] = Q_state_dict[key]*tau + Q_target_state_dict[key]*(1-tau)
        Q_target.load_state_dict(Q_target_state_dict)


def train(env,
          actor,
          Q1,
          Q2,
          Q1_target,
          Q2_target,
          log_alpha,
          actor_optimizer,
          Q1_optimizer,
          Q2_optimizer,
          log_alpha_optimizer,
          memory,
          target_entropy,
          batch_size,
          outdir,
          logdir,
          name,
          gamma=0.99,
          tau=0.005,
          update_after=256,
          updates_per_step=1,
          update_every=1,
          n_episodes=50,
          update_alpha=True,
          dynamic_complexity=True,
          show=True,
          pause_after_episode=False,
          show_live=False,
          pause_after_step=False):
    """
    Train the Soft Actor-Critic (SAC) model.
    
    Parameters
    ----------
    env : object
        The environment to train the model on.
    actor : torch.nn.Module
        The actor network.
    Q1 : torch.nn.Module
        The first Q-value network.
    Q2 : torch.nn.Module
        The second Q-value network.
    Q1_target : torch.nn.Module
        The target network for the first Q-value network.
    Q2_target : torch.nn.Module
        The target network for the second Q-value network.
    log_alpha : torch.Tensor
        The log of the temperature parameter.
    actor_optimizer : torch.optim.Optimizer
        Optimizer for the actor network.
    Q1_optimizer : torch.optim.Optimizer
        Optimizer for the first Q-value network.
    Q2_optimizer : torch.optim.Optimizer
        Optimizer for the second Q-value network.
    log_alpha_optimizer : torch.optim.Optimizer
        Optimizer for the log of the temperature parameter.
    memory : object
        Replay buffer to store transitions.
    target_entropy : float
        The target entropy for the policy.
    batch_size : int
        The batch size for sampling from the replay buffer.
    outdir : str or Path
        Directory to save model snapshots and checkpoints.
    logdir : str or Path
        Directory to save training logs.
    gamma : float
        Discount factor for future rewards.
    tau : float
        Interpolation parameter for target network updates.
    name : str
        Name for saving model snapshots and logs.
    update_after : int, optional
        Number of steps to collect transitions before starting updates (default is 256).
    updates_per_step : int, optional
        Number of updates to perform per step (default is 1).
    update_every : int, optional
        Frequency of updates in terms of steps (default is 1).
    n_episodes : int, optional
        Number of episodes to train for (default is 50).
    dynamic_complexity : bool, optional
        Whether to use dynamic complexity adjustment (default is True).
    show : bool, optional
        Whether to show each finished episode and pause.
    pause_after_episode : bool, optional
        Whether to pause after each episode (default is False).
    show_live : bool, optional
        Whether to show the state of the environment after each step during training.
    pause_after_step : bool, optional
        Whether to pause after each step (default is False).
    """
    # TODO: get a warm start by adding transitions to memory from ground truth streamlines.
    # This could be done by running the environment with an oracle that follows the ground truth streamlines and adding those transitions to the replay buffer before training starts.
    # This would help the agent learn from good examples and potentially speed up training, especially in the early stages when it is mostly taking random actions.

    COMPLEXITY_INCREASE_FREQUENCY = 300  # Incease complexity after this many episodes
    COMPLEXITY_INCREMENT = 0.1  # Amount to increase complexity by
    SAVE_GIF_FREQUENCY = 100  # Save GIF after this many episodes

    steps_done = 0
    last_save = 0
    ep_returns = []
    branching = env.branching
    
    reward_cache = deque(maxlen=10000)
    moving_avg_reward = 0.0

    # Ensure outdir and logdir are Path objects
    outdir = Path(outdir)
    logdir = Path(logdir)
    
    # Create directories if they don't exist
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    
    if show:
        fig, ax = plt.subplots(2, 3, figsize=(15,10))
        plt.ion()

    def _reset_env_and_get_obs():
        """Reset environment and retrieve initial observation with compatibility fallback."""
        try:
            obs0 = env.reset(return_state=True)
            if obs0 is not None:
                return obs0
        except TypeError:
            pass
        env.reset()
        return env.get_state()

    # Train the Network
    # Detect if running in a terminal or redirected (like with nohup)
    use_progress_bar = sys.stdout.isatty()
    policy_device = next(actor.parameters()).device
    for ep in tqdm(range(n_episodes), dynamic_ncols=True, leave=True, file=sys.stdout, mininterval=1.0, disable=not use_progress_bar):
        # Print episode progress when tqdm is disabled (e.g., with nohup)
        if not use_progress_bar:
            print(f"Starting episode {ep + 1}/{n_episodes}", flush=True)
        
        obs = _reset_env_and_get_obs()
        policy_loss = []
        ep_variance = []
        ep_rewards = []
        ep_return = 0
        for t in count():
            # Determine if we have enough samples to start learning updates
            learning_started = steps_done >= update_after
            env.branching = learning_started * branching # only enable branching after learning starts to prevent runaway growth of branches before the agent has learned anything.
            
            if not learning_started:
                action_for_env = torch.randn(3, dtype=torch.float32, device=obs.device) * 3
            else:
                with torch.no_grad():
                    actor_out = actor(obs.to(device=policy_device, dtype=dtype))
                    direction_dist = sample_from_output(actor_out)
                    sampled_action = direction_dist.rsample()[0]
                    var = direction_dist.covariance_matrix.diagonal(dim1=-2, dim2=-1).mean().item()
                action_for_env = sampled_action.to(device=obs.device)
                ep_variance.append(var)
            steps_done += 1
            # take step, get observation and reward, and move index to next streamline
            next_obs, reward, terminated, truncated, info = env.step(action_for_env)
            current_target_vectors = info['current_target_vectors']
            next_target_vectors = info['next_target_vectors'] # this will be None only if path terminates
            if next_target_vectors is None:
                next_target_vectors = current_target_vectors

            reward_value = float(reward.item())
            ep_rewards.append(reward_value)
            ep_return += reward_value

            # Show state after every step
            if show_live:
                try:
                    shell = get_ipython().__class__.__name__  # type: ignore
                    if shell:
                        show_state(env, fig, live=True, reward=reward_value)
                    if pause_after_step:
                        try:
                            print("Press Enter to continue to the next step (or 'q' to quit)...", flush=True)
                            user_input = input().strip()
                            if user_input.lower() == 'q':
                                print("Quitting training.", flush=True)
                                return
                        except EOFError:
                            pass
                except NameError:
                    pass

            reward_cache.append(reward_value)
            moving_avg_reward = sum(reward_cache) / len(reward_cache)

            # Store the transition in memory
            memory.push(obs, action_for_env, next_obs, reward, current_target_vectors, next_target_vectors, terminated)
            
            if learning_started and steps_done % update_every == 0:
                # Perform updates once there is sufficient transitions saved.
                for j in range(updates_per_step):
                    if isinstance(memory, ReplayBuffer):
                        (
                            batch_obs,
                            batch_actions,
                            batch_next_obs,
                            batch_rewards,
                            batch_target_vecs,
                            batch_target_masks,
                            batch_next_target_vecs,
                            batch_next_target_masks,
                            batch_dones,
                        ) = memory.sample(batch_size, transform=True)
                        if gamma > 0: # skip Q updates if gamma is 0
                            td_error = update_Q(actor, Q1, Q1_target, Q2, Q2_target,
                                                batch_obs, batch_actions, batch_next_target_vecs, batch_next_target_masks,
                                                batch_next_obs, batch_dones, Q1_optimizer,
                                                Q2_optimizer, gamma, log_alpha, weights=None)
                            target_update(Q1, Q2, Q1_target, Q2_target, tau)

                    elif isinstance(memory, PrioritizedReplayBuffer):
                        (
                            batch_obs,
                            batch_actions,
                            batch_next_obs,
                            batch_rewards,
                            batch_target_vecs,
                            batch_target_masks,
                            batch_next_target_vecs,
                            batch_next_target_masks,
                            batch_dones,
                            weights,
                            tree_idxs,
                        ) = memory.sample(batch_size, transform=True)
                        if gamma > 0:
                            td_error = update_Q(actor, Q1, Q1_target, Q2, Q2_target,
                                                batch_obs, batch_actions, batch_next_target_vecs, batch_next_target_masks,
                                                batch_next_obs, batch_dones, Q1_optimizer,
                                                Q2_optimizer, gamma, log_alpha, weights=weights)
                            target_update(Q1, Q2, Q1_target, Q2_target, tau)
                            memory.update_priorities(tree_idxs, td_error.cpu().numpy())                    
                    else:
                        raise RuntimeError("Unknown memory buffer")
                        
                    # Perform one step of optimization on the policy network
                    loss = update_actor(batch_obs, batch_target_vecs, batch_target_masks, batch_dones, gamma,
                                        actor, actor_optimizer, Q1, Q2, log_alpha,
                                        log_alpha_optimizer, target_entropy, update_alpha=update_alpha)
                    policy_loss.append(loss)
                    
                    #TODO: Should we update priorities in memory based on reward if not using TD error?
                    # if gamma == 0:
                    #     if isinstance(memory, PrioritizedReplayBuffer):
                    #         memory.update_priorities(tree_idxs, batch_rewards.cpu().numpy())

                    
            if info["terminate_episode"]:
                ep_returns.append(ep_return)
                
                if dynamic_complexity:
                    if ep > 0 and ep % COMPLEXITY_INCREASE_FREQUENCY == 0: 
                        # current_complexity = env.dataloader.complexity
                        # current_morphology = env.dataloader.morphology
                        current_complexity = env.dataset.alpha
                        new_complexity = min(current_complexity + COMPLEXITY_INCREMENT, 1.0)  # Cap complexity at 1.0
                        print(f"Increasing complexity to {new_complexity:.2f}", flush=True)
                        env.dataset.alpha = new_complexity
                        if new_complexity >= 0.3:
                            print("Enabling branching in environment.", flush=True)
                            env.branching = True
                        # First increase morphology filter if not at "any", then increase complexity.
                        # if current_morphology != "any":
                        #     next_morphology = {"simple": "moderate", "moderate": "complex", "complex": "any"}[current_morphology]
                        #     num_images_with_morophology = env.dataloader.dataset.get_complexity_distribution()['morphology_distribution'].get(next_morphology, 0)
                        #     if next_morphology != "any" and num_images_with_morophology < 100:
                        #         print(f"Not enough images with morphology '{next_morphology}' ({num_images_with_morophology} found). Setting morphology to 'any' instead.", flush=True)
                        #         next_morphology = "any"
                        #     env.dataloader.set_morphology(next_morphology)
                        #     print(f"Setting morphology filter to: {next_morphology}", flush=True)
                        #     if next_morphology == "moderate":
                        #         print("Enabling branching in environment.", flush=True)
                        #         env.branching = True
                        # elif current_complexity < 1.0:
                        #     new_complexity = min(current_complexity + COMPLEXITY_INCREMENT, 1.0)  # Cap complexity at 1.0
                        #     print(f"Increasing complexity to {new_complexity:.2f}", flush=True)
                        #     env.dataloader.set_complexity(new_complexity)
                
                if len(policy_loss) > 0:
                    episode_avg_loss = sum(policy_loss)/len(policy_loss) 
                else:
                    episode_avg_loss = 0
                if show:
                    try:
                        shell = get_ipython().__class__.__name__ # type: ignore
                        if shell:
                            show_state(env, fig)
                            print(f"num branches: {len(env.finished_paths)}", flush=True)
                            print(f"reward min/max: {np.min(reward_cache):.2f}/{np.max(reward_cache):.2f} moving avg: {moving_avg_reward:.2f}", flush=True)
                            if pause_after_episode:
                                try:
                                    print("Press Enter to continue to the next episode (or 'q' to quit)...", flush=True)
                                    user_input = input().strip()
                                    if user_input.lower() == 'q':
                                        print("Quitting training.", flush=True)
                                        return
                                except EOFError:
                                    pass
                    except NameError:
                        csv_file_path = logdir / f"{name}_{date}_log.csv"
                        file_exists = csv_file_path.exists()
                        with open(csv_file_path, "a", newline='') as f:
                            writer = csv.writer(f)
                            if not file_exists:
                                writer.writerow(['episode', 'image_file', 'num_branches', 'episode_avg_reward', 'episode_avg_var', 'episode_return', 'episode_avg_policy_loss', 'moving_avg_reward', 'complexity'])
                            num_roots = int(env.roots.shape[0])
                            num_branches = len(env.finished_paths) - num_roots
                            writer.writerow([
                                ep,
                                env.current_neuron_info["neuron_name"],
                                num_branches,
                                np.mean(ep_rewards) if ep_rewards else 0,
                                np.mean(ep_variance) if ep_variance else 0,
                                ep_return,
                                episode_avg_loss,
                                moving_avg_reward,
                                env.dataset.alpha
                            ])
                        if ep % SAVE_GIF_FREQUENCY == 0:
                            trace_gif(env.img.data[:-1].cpu(), env.finished_paths, step_width=env.step_width,
                                    output_path=logdir / f"{name}_{date}_gifs/{name}_{date}_episode_{ep}_image_{env.current_neuron_info['neuron_name'].split('/')[-1].split('.')[0]}_trace.gif",
                                    n_frames=100)                                                                            
                break

            # if episode does not terminate, move to the next state
            if not terminated:
                obs = next_obs
            else:
                # if the path terminated, the observation is zeroed,
                # so we need to get the new observation from the environment
                # which will be the head of the next path.
                obs = env.get_state() 
        
        # save model after at least 500 steps 
        if steps_done // 500 > last_save:
            model_dicts = {
            "policy_state_dict": actor.state_dict(),
            "Q1_state_dict": Q1.state_dict(),
            "Q2_state_dict": Q2.state_dict(),
            "Q1_target_state_dict": Q1_target.state_dict(),
            "Q2_target_state_dict": Q2_target.state_dict(),
            "actor_optimizer_state_dict": actor_optimizer.state_dict(),
            "Q1_optimizer_state_dict": Q1_optimizer.state_dict(),
            "Q2_optimizer_state_dict": Q2_optimizer.state_dict(),
            "log_alpha": log_alpha,
            "log_alpha_optimizer_state_dict": log_alpha_optimizer.state_dict(),
            "steps_done": steps_done
            }
            torch.save(model_dicts, outdir / f"model_state_dicts_{name}_{date}.pt")
            last_save = steps_done // 500

    return
