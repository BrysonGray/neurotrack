import argparse
from datetime import datetime
from itertools import count
import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import sys
import torch
from torch.optim.adamw import AdamW
from torch.optim.adam import Adam
from tqdm import tqdm

sys.path.append(str(Path(__file__).parents[1]))
from environments.sac_tracking_env import Environment
from memory.buffer import ReplayBuffer, PrioritizedReplayBuffer
from plot.show_state import show_state
from models.resblock import ResidualBlock
from models.resnet import ResNet
from models.cnn import ConvNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
date = datetime.now().strftime("%m-%d-%y")


def sample_from_output(out, random=False):    
    '''
    A function to differentiably sample
    '''
    
    mean = out[:,:3] # component 0, 1 and 2
    logvar = out[:,3:] # logvar component 3
    # I want the mean and logvar to be squashed
    # note it would probably be better to not squash components independently
    # better to squash them by their vector norm
    # but it's not a big deal
    meannorm = torch.linalg.norm(mean, dim=-1, keepdim=True)
    meannorm_ = torch.tanh(meannorm)*10 # maximum of 10
    mean = mean * meannorm_/(meannorm + torch.finfo(torch.float).eps)
    logvar = torch.tanh(logvar)*3 + 1 # no very low variance (std is order of 1 pixel) 
    # between 0 and 4
    # std between 1 and 7.4
    # sample = (torch.randn(out.shape[0],3).to(device=out.device)*torch.exp(logvar*0.5) + mean)
    # sample = sample * torch.tensor([[0,1,1]], device=out.device)
    # direction_dist = torch.distributions.MultivariateNormal(mean[:,:2], torch.exp(logvar)[:,None]*torch.eye(2, device=DEVICE)[None])# for paths constrained to a 2d slice
    direction_dist = torch.distributions.MultivariateNormal(mean[:,:3], torch.exp(logvar)[:,None]*torch.eye(3, device=DEVICE)[None])

    return direction_dist


def update_Q(actor, Q1, Q1_target, Q2, Q2_target,
             obs, actions, rewards, next_obs, dones,
             Q1_optimizer, Q2_optimizer, gamma,
             log_alpha, weights=None):

    # Perform one step of the optimization on the Q networks.
    # compute targets
    with torch.no_grad():
        # sample next actions from the current policy
        actor_out = actor(next_obs) # set steps_done to start_steps so that this samples from the current policy
        direction_dist = sample_from_output(actor_out)
        next_directions = direction_dist.rsample()
        logprobs = direction_dist.log_prob(next_directions)
        # next_directions = torch.concatenate((torch.zeros(next_directions.shape[0],1, device=DEVICE), next_directions), dim=1) # for paths constrained to a 2d slice
        # get target q-values
        next_states = torch.concatenate((next_obs, torch.ones((next_obs.shape[0], 1, next_obs.shape[2], next_obs.shape[3], next_obs.shape[4]), 
                                            device=DEVICE)*next_directions[:,:,None,None,None]), dim=1)
        Q1_target_vals = Q1_target(next_states) # vector of q-values for each choice
        Q2_target_vals = Q2_target(next_states)
        targets = rewards + gamma * torch.logical_not(dones) * (torch.minimum(Q1_target_vals, Q2_target_vals) - log_alpha.exp() * logprobs[:,None])
    # compute q-values to compare against targets
    current_state = torch.concatenate((obs, 
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
    return td_error


def update_actor(obs,
                 actor,
                 actor_optimizer,
                 Q1,
                 Q2,
                 log_alpha,
                 log_alpha_optimizer,
                 target_entropy):

    actor_out = actor(obs)
    direction_dist = sample_from_output(actor_out, random=False)
    directions = direction_dist.rsample()
    logprobs = direction_dist.log_prob(directions)
    # get expected Q-vals
    current_state = torch.concatenate((obs, torch.ones((obs.shape[0], 1, obs.shape[2], obs.shape[3], obs.shape[4]),device=DEVICE)*directions[:,:,None,None,None]), dim=1)
    Q1_vals = Q1(current_state)[:,0]
    Q2_vals = Q2(current_state)[:,0]
    # entropy regularized Q values
    loss = -torch.mean(torch.minimum(Q1_vals, Q2_vals) - log_alpha.exp().detach() * logprobs[:,None]) # The loss function is multiplied by -1 to do gradient ascent instead of decent.
    actor_optimizer.zero_grad()
    loss.backward()
    actor_optimizer.step()

    log_alpha_optimizer.zero_grad()
    alpha_loss = (log_alpha.exp() * (-logprobs - target_entropy).detach()).mean()
    alpha_loss.backward()
    log_alpha_optimizer.step()

    return loss.item()


def target_update(Q1, Q2, Q1_target, Q2_target, tau):
    # update target networks
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
          name,
          show_states=True,
          save_snapshots=False,
          update_after=256,
          updates_per_step=1,
          update_every=1,
          n_episodes=50,
          n_trials=1,):
    
    steps_done = 0
    last_save = 0
    ep_returns = []

    if not os.path.isdir(outdir):
        os.makedirs(outdir, exist_ok=True)
    if show_states:
        fig1, ax = plt.subplots(3,3, figure=plt.figure(num=1))
        plt.ion()

    # Train the Network
    for ep in tqdm(range(n_episodes)):
        policy_loss = []
        coverages = []
        trial_returns = []
        labeled_neurons = []
        # run n trials per episode before moving on to the next image.
        # The best trial is saved if save_snapshots==True.
        for trial in range(n_trials):
            obs = env.get_state()
            ep_return = 0
            ep_rewards = []
            for t in count():
                actor.eval()
                if steps_done < update_after:
                    action = torch.randn(3).to(DEVICE)*3
                else:
                    actor_out = actor(obs).detach()
                    direction_dist = sample_from_output(actor_out)
                    action = direction_dist.rsample()[0]
                    
                steps_done += 1
                # take step, get observation and reward, and move index to next streamline
                next_obs, reward, terminated = env.step(action)

                ep_return += reward.cpu().item()
                ep_rewards.append(reward.cpu().item())

                # Store the transition in memory
                memory.push(obs.cpu(), action.cpu(), next_obs.cpu(), reward.cpu(), terminated)
                
                if steps_done >= update_after:
                    if steps_done % update_every == 0:
                        # Perform updates once there is sufficient transitions saved.
                        actor.train()
                        for j in range(updates_per_step):
                            if isinstance(memory, ReplayBuffer):
                                obs, actions, next_obs, rewards, dones = memory.sample(batch_size)
                                td_error = update_Q(actor, Q1, Q1_target, Q2, Q2_target,
                                                    obs, actions, rewards, next_obs, dones,
                                                    Q1_optimizer, Q2_optimizer, gamma,
                                                    log_alpha, weights=None)
                            elif isinstance(memory, PrioritizedReplayBuffer):
                                obs, actions, next_obs, rewards, dones, weights, tree_idxs = memory.sample(batch_size)
                                td_error = update_Q(actor, Q1, Q1_target, Q2, Q2_target,
                                                    obs, actions, rewards, next_obs, dones,
                                                    Q1_optimizer, Q2_optimizer, gamma,
                                                    log_alpha, weights=weights)
                                memory.update_priorities(tree_idxs, td_error.cpu().numpy())                    
                            else:
                                raise RuntimeError("Unknown memory buffer")
                                
                            # Perform one step of optimization on the policy network
                            loss = update_actor(obs, actor, actor_optimizer, Q1, Q2, log_alpha,
                                         log_alpha_optimizer, target_entropy)
                            policy_loss.append(loss)
                            # update target networks
                            target_update(Q1, Q2, Q1_target, Q2_target, tau)

                        
                if terminated:
                    if save_snapshots:
                        labeled_neuron = env.img.data[3].detach().cpu() > 0.3 
                        true_neuron = torch.linalg.norm(env.true_density.data[:3].detach().cpu(), dim=0) > 0.94
                        TP = torch.sum(torch.logical_and(labeled_neuron, true_neuron))
                        tot = torch.sum(true_neuron)
                        coverages.append(TP/tot)
                        labeled_neurons.append(env.img.data[3].detach().clone().cpu())
                        trial_returns.append(ep_return)
                    
                    ep_returns.append(ep_return)
                    if len(policy_loss) > 0:
                        episode_avg_loss = sum(policy_loss)/len(policy_loss) 
                    else:
                        episode_avg_loss = 0
                    if show_states:
                        try:
                            shell = get_ipython().__class__.__name__ # type: ignore
                            if shell:
                                show_state(env, ep_returns, ep_rewards, policy_loss, fig1)
                                print(f"num branches: {len(env.finished_paths)}")
                        except NameError:
                            with open(os.path.join(outdir, f"{name}_{date}_log.txt"), "a") as f:
                                f.write(f"episode: {ep},\n")
                                f.write(f"image file: {env.img_files[env.img_idx].split('/')[-1]}\n")
                                f.write(f"num branches: {len(env.finished_paths)},\n")
                                f.write(f"episode return: {ep_return},\n")
                                f.write(f"episode avg. policy loss: {episode_avg_loss}\n\n")
                    env.reset(move_to_next=False)
                    break
            
                # if not terminated, move to the next state
                obs = env.get_state() # the head of the next streamline
        
        if save_snapshots:
            value, index = torch.max(torch.stack(coverages), dim=0)
            coverage = value.item()
            index = int(index)
            best_return = torch.tensor(trial_returns)[index].item()
            labeled_neuron = labeled_neurons[index]
            paths_dir = os.path.join(outdir, f"episode_snapshots_{name}_{date}/")
            if not os.path.exists(paths_dir):
                os.makedirs(paths_dir)

            to_save = {"labeled_neuron": labeled_neuron, "true_neuron": env.true_density.data[0].detach().clone().cpu(), "coverage": coverage, "return": best_return}
            torch.save(to_save, os.path.join(paths_dir, f"ep_snapshot_{ep}.pt"))

        # save model every 500 steps
        if steps_done // 500 > last_save:
            model_dicts = {"policy_state_dict": actor.state_dict(),
                        "Q1_state_dict": Q1_target.state_dict(),
                        "Q2_state_dict": Q2_target.state_dict(),}
            torch.save(model_dicts, os.path.join(outdir, f"model_state_dicts_{name}_{date}.pt"))
            last_save = steps_done // 500
        env.reset()

    return


