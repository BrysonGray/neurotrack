
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
import select
import sys
import torch
from tqdm import tqdm

script_path = Path(os.path.abspath(__file__))
parent_dir = script_path.parent.parent
sys.path.append(str(parent_dir))
from memory.buffer import ReplayBuffer, PrioritizedReplayBuffer
from plot.tracking_interface import show_state
import ipywidgets as widgets
from IPython.display import display

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
date = datetime.now().strftime("%m-%d-%y")


def sample_from_output(out, random=False):    
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
    mean = mean * meannorm_/(meannorm + torch.finfo(torch.float).eps)
    logvar = torch.tanh(logvar)*3 + 1 # no very low variance (std is order of 1 pixel) 
    direction_dist = torch.distributions.MultivariateNormal(mean[:,:3], torch.exp(logvar)[:,None]*torch.eye(3, device=out.device)[None])

    return direction_dist


def update_Q(actor, Q1, Q1_target, Q2, Q2_target,
             obs, actions, rewards, next_obs, dones,
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
    rewards : torch.Tensor
        The rewards received.
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
        actor_out = actor(next_obs) # set steps_done to start_steps so that this samples from the current policy
        direction_dist = sample_from_output(actor_out.detach().cpu())
        next_directions = direction_dist.rsample()
        logprobs = direction_dist.log_prob(next_directions)
        next_directions = next_directions.to(DEVICE)
        logprobs = logprobs.to(DEVICE)
        # get target q-values
        next_states = torch.cat((next_obs, torch.ones((next_obs.shape[0], 1, next_obs.shape[2], next_obs.shape[3], next_obs.shape[4]), 
                                            device=DEVICE)*next_directions[:,:,None,None,None]), dim=1)
        Q1_target_vals = Q1_target(next_states) # vector of q-values for each choice
        Q2_target_vals = Q2_target(next_states)
        targets = rewards + gamma * torch.logical_not(dones) * (torch.minimum(Q1_target_vals, Q2_target_vals) - log_alpha.exp() * logprobs[:,None])
        
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
    current_state = torch.cat((obs, 
                        torch.ones((obs.shape[0], 1, obs.shape[2], obs.shape[3], obs.shape[4]), 
                                    device=DEVICE)*actions[:,:,None,None,None]), dim=1)
    
    if weights is None:
        weights = torch.ones_like(targets, device=DEVICE)
    
    if weights.device != DEVICE:
        weights = weights.to(device=DEVICE)

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


def update_actor(obs, actor, actor_optimizer, Q1, Q2, log_alpha, log_alpha_optimizer, target_entropy):
    """
    Update the actor network and the temperature parameter in the Soft Actor-Critic (SAC) algorithm.

    Parameters
    ----------
    obs : torch.Tensor
        The observations from the environment.
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
    # direction_dist = sample_from_output(actor_out.detach().cpu(), random=False)
    direction_dist = sample_from_output(actor_out, random=False)
    directions = direction_dist.rsample()
    logprobs = direction_dist.log_prob(directions)
    directions = directions.to(DEVICE)
    logprobs = logprobs.to(DEVICE)
    # get expected Q-vals
    current_state = torch.cat((obs, torch.ones((obs.shape[0], 1, obs.shape[2], obs.shape[3], obs.shape[4]),device=DEVICE)*directions[:,:,None,None,None]), dim=1)
    Q1_vals = Q1(current_state)[:,0]
    Q2_vals = Q2(current_state)[:,0]
    # entropy regularized Q values
    loss = -torch.mean(torch.minimum(Q1_vals, Q2_vals) - log_alpha.exp().detach() * logprobs[:,None]) # The loss function is multiplied by -1 to do gradient ascent instead of decent.
    actor_optimizer.zero_grad()
    loss.backward()
    actor_optimizer.step()

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
          gamma,
          tau,
          outdir,
          logdir,
          name,
          update_after=256,
          updates_per_step=1,
          update_every=1,
          n_episodes=50,
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
    gamma : float
        Discount factor for future rewards.
    tau : float
        Soft update parameter for updating the target networks.
    outdir : str or Path
        Directory to save model snapshots and checkpoints.
    logdir : str or Path
        Directory to save training logs.
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

    steps_done = 0
    last_save = 0
    ep_returns = []
    
    reward_cache = deque(maxlen=10000)
    moving_avg_reward = 0.0
    if dynamic_complexity:
        last_avg_reward = 0.0  # Track previous average for improvement detection

    # Ensure outdir and logdir are Path objects
    outdir = Path(outdir)
    logdir = Path(logdir)
    
    # Create directories if they don't exist
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)
    
    if show:
        fig, ax = plt.subplots(2, 3, figsize=(15,10))
        plt.ion()

    # Train the Network
    # Detect if running in a terminal or redirected (like with nohup)
    use_progress_bar = sys.stdout.isatty()
    for ep in tqdm(range(n_episodes), dynamic_ncols=True, leave=True, file=sys.stdout, mininterval=1.0, disable=not use_progress_bar):
        # Print episode progress when tqdm is disabled (e.g., with nohup)
        if not use_progress_bar:
            print(f"Starting episode {ep + 1}/{n_episodes}", flush=True)
        
        env.reset()
        policy_loss = []
        obs = env.get_state()
        ep_return = 0
        for t in count():
            # Determine if we have enough samples to start learning updates
            learning_started = steps_done >= update_after
            
            if not learning_started:
                action = torch.randn(3)*3
            else:
                actor_out = actor(obs.to(DEVICE))
                direction_dist = sample_from_output(actor_out.detach().cpu())
                action = direction_dist.rsample()[0]
                
            steps_done += 1
            # take step, get observation and reward, and move index to next streamline
            next_obs, reward, terminated, truncated, info = env.step(action, training=learning_started)
            target_vector = info['target_vector']

            ep_return += reward.cpu().item()

            # Show state after every step
            if show_live:
                try:
                    shell = get_ipython().__class__.__name__  # type: ignore
                    if shell:
                        show_state(env, fig, live=True, reward=reward.cpu().item())
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

            reward_cache.append(reward.cpu().item())
            moving_avg_reward = sum(reward_cache) / len(reward_cache)

            # Store the transition in memory
            memory.push(obs.cpu(), action.cpu(), next_obs.cpu(), reward.cpu(), target_vector, terminated)
            
            if learning_started and steps_done % update_every == 0:
                # Perform updates once there is sufficient transitions saved.
                for j in range(updates_per_step):
                    if isinstance(memory, ReplayBuffer):
                        batch_obs, batch_actions, batch_next_obs, batch_rewards, batch_target_vecs, batch_dones = memory.sample(batch_size, transform=True)
                        td_error = update_Q(actor, Q1, Q1_target, Q2, Q2_target,
                                            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones,
                                            Q1_optimizer, Q2_optimizer, gamma,
                                            log_alpha, weights=None)
                    elif isinstance(memory, PrioritizedReplayBuffer):
                        batch_obs, batch_actions, batch_next_obs, batch_rewards, batch_target_vecs, batch_dones, weights, tree_idxs = memory.sample(batch_size, transform=True)
                        td_error = update_Q(actor, Q1, Q1_target, Q2, Q2_target,
                                            batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones,
                                            Q1_optimizer, Q2_optimizer, gamma,
                                            log_alpha, weights=weights)
                        memory.update_priorities(tree_idxs, td_error.cpu().numpy())                    
                    else:
                        raise RuntimeError("Unknown memory buffer")
                        
                    # Perform one step of optimization on the policy network
                    loss = update_actor(batch_obs, actor, actor_optimizer, Q1, Q2, log_alpha,
                                 log_alpha_optimizer, target_entropy)
                    policy_loss.append(loss)
                    # update target networks
                    target_update(Q1, Q2, Q1_target, Q2_target, tau)

                    
            if info["terminate_episode"]:
                ep_returns.append(ep_return)
                
                if dynamic_complexity:
                    # Check if model is improving based on recent step rewards (at least 50 rewards for stable comparison)
                    if ep >= 50: # start updating complexity after 50 episodes
                        improvement = moving_avg_reward > max(0.0, last_avg_reward)
                        if improvement:
                            current_complexity = env.dataloader.complexity
                            if current_complexity < 1.0:
                                new_complexity = min(current_complexity + 0.05, 1.0)  # Cap complexity at 1.0
                                print(f"Improvement detected. Increasing complexity to {new_complexity:.2f}", flush=True)
                                env.dataloader.set_complexity(new_complexity)
                                if new_complexity > 0.33 and env.branching == False:
                                    print("Enabling branching in environment.", flush=True)
                                    env.branching = True
                        # Update the last average for next comparison
                        last_avg_reward = moving_avg_reward
                
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
                                writer.writerow(['episode', 'image_file', 'num_branches', 'episode_return', 'episode_avg_policy_loss', 'moving_avg_reward'])
                            writer.writerow([
                                ep,
                                env.current_neuron_info["neuron_name"],
                                len(env.finished_paths),
                                ep_return,
                                episode_avg_loss,
                                moving_avg_reward
                            ])
                break

            # if episode does not terminate, move to the next state
            obs = env.get_state() # the head of the next streamline
        
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


def inference(env, actor, outdir, Q_net=None, n_trials=1, show=True, show_live=True, save_paths=False, sync=False, stochastic=False):
    """
    Perform inference using the given actor in the specified environment.
    
    Parameters
    ----------
    env : object
        The environment object which provides the state and handles the actions.
    actor : object
        The actor model used to perform actions in the environment.
    outdir : str
        The directory where the inference results will be saved.
    n_trials : int, optional
        The number of trials to perform for each image (default is 5).
    show : bool, optional
        If True, display the state of the environment during inference (default is True).
        
    Returns
    -------
    None
    """

    if show:
        fig, ax = plt.subplots(2, 3, figsize=(15,10))
        plt.ion()
    actor.eval()
    if sync:
        # find image indices that are not yet processed
        # get a list of image names that are already processed
        processed_image_names = [re.split(r'_\d\d-\d\d-\d\d_inference.npz', f)[0] for f in os.listdir(outdir) if f.endswith('.npz')]
        image_names = [entry["neuron_name"] for entry in env.dataloader.dataset]
        img_indices = [i for i, f in enumerate(image_names) if f.split('/')[-1] not in processed_image_names]
    else:
        img_indices = [i for i in range(env.dataloader.current_idx, len(env.dataloader.dataset))]

    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    elif n_trials > 1 and Q_net is None:
        raise ValueError("Q_net must be provided if n_trials > 1")

    # Detect if running in a terminal or redirected (like with nohup)
    use_progress_bar = sys.stdout.isatty()
    for i in tqdm(range(len(img_indices)), dynamic_ncols=True, leave=True, file=sys.stdout, mininterval=1.0, disable=not use_progress_bar):
        # Print inference progress when tqdm is disabled (e.g., with nohup)
        if not use_progress_bar:
            print(f"Processing image {i + 1}/{len(img_indices)}", flush=True)
            
        img_idx = img_indices[i] % len(env.dataloader.dataset) # -1 because the index is incremented when the environment resets
        env.reset(dataset_index=img_idx)
        coverages = []
        estimated_returns = []
        labeled_neurons = []
        trial_paths = []
        # Begin trials. Loop through n_trials.
        for trial in range(n_trials):
            estimated_return = 0
            obs = env.get_state()
            # Begin episode. Loop through steps.
            for t in count():
                with torch.no_grad():
                    actor_out = actor(obs.to(DEVICE))
                    direction_dist = sample_from_output(actor_out.detach().cpu())
                    if stochastic:
                        action = direction_dist.sample()[0]
                    else:
                        action = direction_dist.mean[0]

                next_obs, reward, terminated, truncated, info = env.step(action, training=True)
                if Q_net is not None:
                    current_state = torch.cat((obs.to(DEVICE), torch.ones((obs.shape[0], 1, obs.shape[2], obs.shape[3], obs.shape[4]),device=DEVICE)*action[None,:,None,None,None].to(DEVICE)), dim=1)
                    q_val = Q_net(current_state)[:,0]
                    estimated_return += q_val.cpu().item()

                # Show state after every step
                if show_live and show:
                    try:
                        shell = get_ipython().__class__.__name__  # type: ignore
                        if shell:
                            show_state(env, fig, live=True, reward=reward.cpu().item())
                    except NameError:
                        pass

                if info["terminate_episode"]:
                    labeled_neuron = env.img.data[-1].detach().cpu() > 0.3 
                    # true_neuron = env.true_density.data[-1].detach().cpu() > 0.94
                    # TP = torch.sum(torch.logical_and(labeled_neuron, true_neuron))
                    # tot = torch.sum(true_neuron)
                    # coverages.append(TP/tot)
                    estimated_returns.append(estimated_return)
                    labeled_neurons.append(env.img.data[-1].detach().clone().cpu())
                    trial_paths.append([path.detach().cpu().numpy().tolist() for path in env.finished_paths if isinstance(path, torch.Tensor) and len(path) > 3])
                    if show:
                        try:
                            shell = get_ipython().__class__.__name__ # type: ignore
                            if shell:
                                # Show final state and episode info, then wait for user input
                                show_state(env, fig)
                                print(f"num branches: {len(env.finished_paths)}")
                                print(f"num long paths: {len(trial_paths[-1])}")
                                if Q_net is not None:
                                    print(f"estimated return: {estimated_return:.2f}")
                                # print(f"coverage: {coverages[-1]:.2f}")
                                try:
                                    print("Press Enter to continue to the next episode (or 'q' to quit)...", flush=True)
                                    user_input = input().strip()
                                    if user_input.lower() == 'q':
                                        print("Quitting training.", flush=True)
                                        return
                                except EOFError:
                                    pass       
                        except NameError:
                            pass
                    env.reset(move_to_next=False)
                    break # Move to next trial. (same seed, same image)

                obs = env.get_state()

        if save_paths:
            value = np.max(estimated_returns)
            index = np.argmax(estimated_returns)
            estimated_return = value.item()
            index = int(index)
            labeled_neuron = labeled_neurons[index]
            name = env.current_neuron_info["neuron_name"]
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            # Convert paths to a serializable format
            paths_to_save = trial_paths[index]
            labeled_neuron_np = labeled_neuron.numpy()

            # Using numpy's compressed format
            np.savez_compressed(os.path.join(outdir, f"{name}_{date}_inference.npz"),
                                labeled_neuron=labeled_neuron_np,
                                coverages=coverages,
                                estimated_returns=estimated_returns,
                                paths=np.array(paths_to_save, dtype=object))

    return

if __name__ == "__main__":
    pass