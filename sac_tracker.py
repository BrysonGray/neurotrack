#!/usr/bin/env python

"""
Soft actor-critic tractography model functions

Author: Bryson Gray
2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
import math
import os
import matplotlib
import matplotlib.pyplot as plt
from collections import deque, namedtuple
from itertools import count
from tqdm import tqdm
from scipy.stats import vonmises_fisher

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# plotting functions
def plot_durations(episode_durations, show_result=False):
    plt.figure()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            # display.display(plt.gcf())
            display.display(plt.figure(),display_id=1)
            display.clear_output(wait=True)
        else:
            # display.display(plt.gcf(), display_id=True)
            display.display(plt.figure(), display_id=1)


def plot_returns(episode_returns, show_result=False):
    plt.figure()
    return_t = torch.tensor(episode_returns, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Return')
    plt.plot(return_t.numpy())
    # Take 100 episode averages and plot them too
    if len(return_t) >= 100:
        means = return_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            # display.display(plt.gcf())
            display.display(plt.figure(), display_id=2)
            display.clear_output(wait=True)
        else:
            # display.display(plt.gcf())
            display.display(plt.figure(), display_id=2)

# class ReplayMemory():

#     def __init__(self, capacity):
#         self.memory = deque([],maxlen=capacity)
#         self.Transition = namedtuple('Transition',
#                         ('state', 'action', 'next_state', 'reward', 'done'))

#     def push(self, *args):
#         """Save a transition"""
#         self.memory.append(self.Transition(*args))

#     def sample(self, batch_size):
#         return random.sample(self.memory, batch_size)

#     def __len__(self):
#         return len(self.memory)
    
class ReplayMemory():
    def __init__(self, capacity, obs_shape, action_shape):

        self.obs = torch.empty((capacity, *obs_shape), dtype=torch.float32)
        self.last_steps = torch.empty((capacity, 3), dtype=torch.float32)
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32)
        self.next_obs = torch.empty((capacity, *obs_shape), dtype=torch.float32)
        self.next_steps = torch.empty((capacity, 3))
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32)
        self.dones = torch.empty((capacity, 1), dtype=torch.bool)

        self.idx = 0
        self.full = False
    
    def push(self, obs, last_step, action, next_obs, next_step, reward, done):
        """Save a transition to replay memory"""
        self.obs[self.idx] = obs
        self.last_steps[self.idx] = last_step
        self.actions[self.idx] = action
        self.next_obs[self.idx] = next_obs
        self.next_steps[self.idx] = next_step
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.idx == 0 or self.full
    
    def sample(self, batch_size):
        sample_range = self.capacity if self.full else self.idx
        idxs = torch.randint(sample_range, size=batch_size)

        obs = self.obs[idxs]
        last_steps = self.last_steps[idxs]
        actions = self.actions[idxs]
        next_obs = self.next_obs[idxs]
        next_steps = self.next_steps
        rewards = self.rewards[idxs]
        dones = self.dones[idxs]

        return obs, last_steps, actions, next_obs, next_steps, rewards, dones
    
    def __len__(self):
        return self.capacity if self.full else self.idx


class Critic(nn.Module):
    """
    Double Deep Q-Network CNN critic network model. This takes a state and an action as input,
    where the action is a direction in R^3 and an integer choice in {0,1,2} which represent
    step, terminate, and branch, respectively.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    n_actions : int
        Number of actions in the action space
    input_size : int
        Input image size of one dimension. Used to
        calculate the number of input features in the fully connected layer.
        Note the input must have equal height, width, and depth.
    """

    def __init__(self, in_channels, input_size, step_size=torch.tensor([1.0,1.0,1.0]), n=1):
        super().__init__()

        self.step_size = step_size.to(dtype=torch.float32, device=DEVICE)

        # calculate the size of the convolution output for input to Linear
        shape_out = lambda x, k, s: ((x - k)/s + 1)//1
        h = shape_out(input_size, 3, 2) # first conv
        h = shape_out(h, 3, 2) # second conv
        n_features = int(32*n * h**3)

        # convolution layers
        self.conv1 = nn.Conv3d(in_channels + 7, 16*n, 3, stride=2) # 3 channels added to in_channels for components of previous step direction and 4 more for the action
        self.norm1 = nn.BatchNorm3d(16*n)
        self.conv2 = nn.Conv3d(16*n, 32*n, 3, stride=2)
        self.norm2 = nn.BatchNorm3d(32*n)

        # fully connected layers
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, obs, step, action):
        """
        Parameters
        ----------
        obs : torch.Tensor
            Observtion of the image patch with shape (M, 3, L, L, L), where L is the side length of the patch.
        step : torch.Tensor
            Last step direction with shape (M, 3)
        action : torch.Tensor
            The action taken. This has shape (M, 4) where the first 3 components of each
            action is the step direction and the last component is the choice (step, terminate, branch).

        """
        # concatenate previous step direction components
        x = torch.concatenate((obs, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]), device=DEVICE)*step[:,:,None,None,None]), dim=1)
        # concatenate action
        x = torch.concatenate((x, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]), device=DEVICE)*action[:,:,None,None,None]), dim=1)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    

class Actor(nn.Module):

    def __init__(self, in_channels, input_size, step_size=torch.tensor([1.0,1.0,1.0]), n=1):
        super().__init__()
        self.step_size = step_size.to(dtype=torch.float32, device=DEVICE)

        # calculate the size of the convolution output for input to Linear
        shape_out = lambda x, k, s: ((x - k)/s + 1)//1
        h = shape_out(input_size, 3, 2) # first conv
        h = shape_out(h, 3, 2) # second conv
        n_features = int(32*n * h**3)

        # convolution layers
        self.conv1 = nn.Conv3d(in_channels + 3, 16*n, 3, stride=2) # 3 channels added to in_channels for components of previous step direction
        self.norm1 = nn.BatchNorm3d(16*n)
        self.conv2 = nn.Conv3d(16*n, 32*n, 3, stride=2)
        self.norm2 = nn.BatchNorm3d(32*n)

        # fully connected layers
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        # The last fully connected layer outputs three means and standard deviations for the step direction components,
        # plus the three element logits representing the choice between taking a step, terminating the path, or branching.
        self.fc3 = nn.Linear(128, 9)
    
    def forward(self, obs, step):
        # x,p = state
        # w = (p[:,1] - p[:,0]) / self.step_size
        
        x = torch.concatenate((obs, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]), device=DEVICE)*step[:,:,None,None,None]), dim=1)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        mu = x[:3] # the step direction is equal to mu / ||mu||
        sigma = F.relu(x[3:6]) # standard deviation must be positive
        choice = x[6:] # logits

        direction_dist = torch.distributions.Normal(mu, sigma)
        choice_dist = torch.distributions.Categorical(choice)

        return direction_dist, choice_dist


class SACModel():
    def __init__(self, in_channels, input_size, start_steps=1000, lr=0.005, gamma=0.99, entropy_coeff=0.2, step_size=torch.tensor([1.0,1.0,1.0])):
    
        self.actor = Actor(in_channels, input_size, step_size).to(DEVICE)
        self.Q1 = Critic(in_channels, input_size, step_size).to(DEVICE)
        self.Q2 = Critic(in_channels, input_size, step_size).to(DEVICE)
        self.Q1_target = Critic(in_channels, input_size, step_size).to(DEVICE)
        self.Q2_target = Critic(in_channels, input_size, step_size).to(DEVICE)
        # initialize the Q1 and Q2 target networks with the same params as Q1 and Q2
        self.Q1_target.load_state_dict(self.Q1.state_dict()) 
        self.Q2_target.load_state_dict(self.Q2.state_dict())

        self.Q1_optimizer = torch.optim.AdamW(self.Q1.parameters(), lr=lr, amsgrad=True)
        self.Q2_optimizer = torch.optim.AdamW(self.Q2.parameters(), lr=lr, amsgrad=True)
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=lr, amsgrad=True)

        self.lrscheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9993)

        # self.memory = ReplayMemory(10000, )

        self.start_steps = start_steps # random steps taken before beginning to sample from the policy
        self.gamma = gamma # discount factor
        self.entropy_coeff = entropy_coeff


    def load_model(self, actor_weights, Q1_weights, Q2_weights):
        self.Q1.load_state_dict(Q1_weights)
        self.Q2.load_state_dict(Q1_weights)
        self.Q1_target.load_state_dict(Q2_weights)
        self.Q2_target.load_state_dict(Q2_weights)
        self.actor.load_state_dict(actor_weights)


    def select_action(self, obs, last_step, steps_done=0, sample=True):
        if steps_done < self.start_steps:
            direction = torch.randn((3,))
            direction = direction / torch.sum(direction**2)**0.5
            # Do not use a uniform distribution over choices because our prior expectation is it
            # will step much more often than branching or terminating a patch. It should also
            # terminate slightly more than branch otherwise an episode could run indefinitely.
            choice = torch.multinomial(torch.tensor([0.98, 0.024, 0.016]), num_samples=1)
        else:
            direction_dist, categorical_dist = self.actor(obs, last_step)
            if sample:
                direction = direction_dist.rsample()
                choice = categorical_dist.rsample()
            else:
                direction = direction_dist.mean
                choice = torch.argmax(categorical_dist.probs)

            
        return direction, choice


    def optimize_Q(self, obs, step, actions, next_obs, next_step, rewards, dones):
        """ Do one optimization step
        
        """
        # compute targets
        direction_dist, choice_dist = self.select_action(next_obs, next_step)
        next_direction = direction_dist.rsample()
        next_choice = choice_dist.rsample()
        log_prob = direction_dist.log_prob(next_direction).sum(-1, keepdim=True) + choice_dist.log_prob(next_choice)

        action = torch.cat((next_direction, next_choice), dim=-1)
        Q1_target_vals = self.Q1_target(next_obs, next_step, action)
        Q2_target_vals = self.Q2_target(next_obs, next_step, action)
        targets = rewards + self.gamma * torch.logical_not(dones) * (torch.minimum(Q1_target_vals, Q2_target_vals) - self.entropy_coeff * log_prob)

        Q1_vals = self.Q1(obs, step, action)
        Q2_vals = self.Q2(obs, step, action)

        criterion = nn.MSELoss()

        Q1_loss = criterion(Q1_vals, targets)
        Q2_loss = criterion(Q2_vals, targets)

        self.Q1_optimizer.zero_grad()
        self.Q2_optimizer.zero_grad()

        Q1_loss.backward()
        Q2_loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), 1)

        self.Q1_optimizer.step()
        self.Q2_optimizer.step()

        return
    
    def optimize_policy(self, obs, step):

        direction_dist, choice_dist = self.select_action(obs, step)
        direction = direction_dist.rsample()
        choice = choice_dist.rsample()
        action = torch.cat((direction, choice), dim=-1)
        log_prob = direction_dist.log_prob(direction).sum(-1, keepdim=True) + choice_dist.log_prob(choice)

        Q1_vals = self.Q1(obs, step, action)
        Q2_vals = self.Q1(obs, step, action)

        Q_min = torch.minimum(Q1_vals,Q2_vals)

        loss = (self.entropy_coeff.detach() * log_prob - Q_min).mean()

        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return
    

    def train(self, env, episodes=100, batch_size=128, gamma=0.99, tau=0.001, alpha=0.001, eps_start=0.9, eps_end=0.01,
              eps_decay=1000, save_snapshots=True, output='./outputs', name='config0', show=False):

        steps_done = 0
        episode_durations = []
        episode_returns = []
        losses = []
        grad_norms = []
        lr_vals = []
        eps = []
        mae = []
        bending_energy = []
        friction = []
        n_paths = []
        if not os.path.exists(output):
            os.makedirs(output)
        # Train the Network
        for i in tqdm(range(episodes)):
            env.reset()
            state = env.get_state()
            state = (state[0].to(dtype=torch.float32, device=DEVICE),\
                     state[1].to(dtype=torch.float32, device=DEVICE))
            ep_return = 0
            for t in count():
                action = self.select_action(state, steps_done, eps_start, eps_end, eps_decay) # returns an index for action_space
                steps_done += 1
                # take step, get observation and reward, and move index to next streamline
                observation, reward, terminated = env.step(action)
                ep_return += reward

                next_state = observation
                next_state = (next_state[0].to(dtype=torch.float32, device=DEVICE),\
                                next_state[1].to(dtype=torch.float32, device=DEVICE))

                # TODO: change for testing, 6/18
                # if terminated: # episode terminated
                #     next_state = None
                # else:
                #     next_state = observation # if the streamline terminated then observation is None
                #     if next_state is not None:
                #         next_state = (next_state[0].to(dtype=torch.float32, device=DEVICE),\
                #                       next_state[1].to(dtype=torch.float32, device=DEVICE))


                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Perform one step of the optimization (on the policy network)
                loss = self.optimize_model(batch_size, gamma)
                if loss:
                    if steps_done%50 == 0:
                        losses.append(loss)
                        
                        total_norm = 0
                        for p in self.policy_net.parameters():
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** (1. / 2)
                        grad_norms.append(total_norm)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if terminated:
                    episode_durations.append(t + 1)
                    episode_returns.append(ep_return)
                    n_paths.append(len(env.finished_paths))
                    # save global matching error
                    # mae.append(torch.mean(torch.abs(env.bundle_density.data - env.true_density.data)))
                    mae.append(torch.mean(torch.abs(env.img.data[3] - env.true_density.data)))
                    # save global bending energy
                    friction.append(torch.sum(torch.tensor([len(path) for path in env.finished_paths])))
                    bending_energy_ = []
                    for j in range(len(env.finished_paths)):
                        p0 = env.finished_paths[j][:-1]
                        p1 = env.finished_paths[j][1:]
                        segments = (p1 - p0) / env.step_size
                        energy = (torch.einsum('ij,ij->i', segments[1:], segments[:-1]) - 1.0) / -2.0
                        bending_energy_.append(torch.sum(energy))
                    bending_energy.append(torch.sum(torch.tensor(bending_energy_)))
                    # save global friction
                    
                    if show:
                        plot_durations(episode_durations)
                        plot_returns(episode_returns)
                    if save_snapshots:
                        if i%17 == 0:
                            img_out = (env.img.data[3].detach().clone() > 0.0) * 0.4 + env.true_density.data[0].detach().clone() * 0.6
                            # torch.save(env.img.data[3].detach().clone(), os.path.join(output, f'bundle_density_ep{i%6}.pt'))
                            torch.save(img_out, os.path.join(output, f'bundle_density_ep{i%6}.pt'))
                            if env.branching:
                                torch.save(env.img.data[4].detach().clone(), os.path.join(output, f'bifurcations_ep{i%6}.pt'))
                            torch.save(target_net_state_dict, os.path.join(output, f'model_state_dict_{name}.pt'))
                            torch.save(episode_durations, os.path.join(output, f'episode_durations_{name}.pt'))
                            torch.save(episode_returns, os.path.join(output, f'episode_returns_{name}.pt'))
                            torch.save(mae, os.path.join(output, f'matching_error_{name}.pt'))
                            torch.save(bending_energy, os.path.join(output, f'bending_energy_{name}.pt'))
                            torch.save(friction, os.path.join(output, f'friction_{name}.pt'))
                            torch.save(n_paths, os.path.join(output, f'n_paths_{name}.pt'))

                            torch.save(losses, os.path.join(output, f'loss_{name}.pt'))
                            torch.save(grad_norms, os.path.join(output, f'grad_norms_{name}.pt'))
                            torch.save(lr_vals, os.path.join(output, f'lr_{name}.pt'))
                            eps.append(eps_end + (eps_start - eps_end) * math.exp(-1. * steps_done / eps_decay))
                            torch.save(eps, os.path.join(output, f'eps_{name}.pt'))
                    break

                # if not terminated, move to the next state
                state = env.get_state() # the head of the next streamline
                state = (state[0].to(dtype=torch.float32, device=DEVICE),\
                        state[1].to(dtype=torch.float32, device=DEVICE))
                
            if len(self.memory) > batch_size:
                self.lrscheduler.step()
                lr = self.lrscheduler.get_last_lr()
                lr_vals.append(lr)

        print('Complete')
        if show:
            plt.ioff()
            plt.show()

        return
    

    def inference(self, env, out=None):

            env.reset()
            state = env.get_state()
            state = (state[0].to(dtype=torch.float32, device=DEVICE),\
                     state[1].to(dtype=torch.float32, device=DEVICE))
            ep_return = 0
            i = 0
            plt.ioff()
            
            while True:
                # get action
                action = self.select_action(state, greedy=True)

                # take step, get observation and reward, and move index to next streamline
                observation, reward, terminated = env.step(action) 
                ep_return += reward
                i += 1

                if terminated:
                    return env

                else:
                    # if not terminated, move to the next state
                    state = env.get_state() # the head of the next streamline
                    state = (state[0].to(dtype=torch.float32, device=DEVICE),\
                            state[1].to(dtype=torch.float32, device=DEVICE))
                    if out is not None and i%10 == 0:
                        plt.figure(0)
                        plt.imshow(env.img.data[-1].amax(dim=0), cmap='hot', alpha=0.5)#, int(paths[env.head_id][-1, 0])])
                        plt.imshow(env.img.data[:3].amin(dim=1).permute(1,2,0), alpha=0.5)
                        plt.axis('off')
                        plt.savefig(os.path.join(out, f'path_{i}.png'))