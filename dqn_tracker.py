#!/usr/bin/env python

"""
Deep Q-network tractography model functions

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

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# plotting functions
def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
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
            display.display(plt.figure(1),display_id=1)
            display.clear_output(wait=True)
        else:
            # display.display(plt.gcf(), display_id=True)
            display.display(plt.figure(1), display_id=1)


def plot_returns(episode_returns, show_result=False):
    plt.figure(2)
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
            display.display(plt.figure(2), display_id=2)
            display.clear_output(wait=True)
        else:
            # display.display(plt.gcf())
            display.display(plt.figure(2), display_id=2)


class ReplayMemory():

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):
    """
    Deep Q-Network CNN network model.

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

    def __init__(self, in_channels, n_actions, input_size):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, 8, 2)
        self.norm1 = nn.BatchNorm3d(8)
        self.conv2 = nn.Conv3d(8, 16, 5)
        self.norm2 = nn.BatchNorm3d(16)
        # self.pool = nn.MaxPool3d(2, 2)

        # calculate the size of the convolution output for input to Linear
        shape_out = lambda x, k: (x - (k-1) - 1) + 1
        h = shape_out(input_size, 2) # first conv
        h = shape_out(h, 5) # second conv
        n_features = 16 * h**3

        self.fc1 = nn.Linear(n_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_actions)

    def forward(self, x):
        x = self.norm1(F.relu(self.conv1(x)))
        x = self.norm2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x

class DQNModel():
    def __init__(self, in_channels, n_actions, input_size, lr):
    
        self.policy_net = DQN(in_channels, n_actions, input_size).to(DEVICE)
        self.target_net = DQN(in_channels, n_actions, input_size).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self. optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def load_model(self, model_weights):
        self.policy_net.load_state_dict(model_weights)
        self.target_net.load_state_dict(model_weights)

    def select_action(self, action_space, observation, steps_done, eps_start=0.9, eps_end=0.01, eps_decay=1000):
        sample = random.random()
        eps_threshold = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * steps_done / eps_decay)

        if sample > eps_threshold:
            with torch.no_grad():
                #  pick action with the larger expected reward.
                return torch.argmax(self.policy_net(observation))
        else:
            # take a random action  
            return torch.randint(len(action_space), (1,), device=DEVICE).squeeze()


    def optimize_model(self, batch_size, gamma):
        """ Do one optimization step
        
        """
        if len(self.memory) < batch_size:
            return
        transitions = self.memory.sample(batch_size)

        # Convert batch-array of transitions to transition of batch-arrays
        batch = self.memory.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.stack(batch.action).unsqueeze(1)
        reward_batch = torch.cat(batch.reward).to(device=DEVICE)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size, device=DEVICE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, env, episodes=100, batch_size=128, gamma=0.99, tau=0.001, eps_start=0.9, eps_end=0.01,
              eps_decay=1000, save_snapshots=True, output='./outputs', name='config0', show=False):

        steps_done = 0
        episode_durations = []
        episode_returns = []
        if not os.path.exists(output):
            os.makedirs(output)
        # Train the Network
        for i in tqdm(range(episodes)):
            env.reset()
            state = env.get_state()[0].clone().to(dtype=torch.float32, device=DEVICE)
            ep_return = 0
            for t in count():
                action_id = self.select_action(env.action_space, state, steps_done, eps_start, eps_end, eps_decay)
                # take step, get observation and reward, and move index to next streamline
                observation, reward, terminated = env.step(env.action_space[action_id]) 
                ep_return += reward

                if terminated: # episode terminated
                    next_state = None
                else:
                    next_state = observation # if the streamline terminated observation is None
                    if next_state is not None:
                        next_state = next_state[0].clone().to(dtype=torch.float32, device=DEVICE)

                # Store the transition in memory
                self.memory.push(state, action_id, next_state, reward)

                # Perform one step of the optimization (on the policy network)
                self.optimize_model(batch_size, gamma)

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
                    if show:
                        plot_durations(episode_durations)
                        plot_returns(episode_returns)
                    if save_snapshots:
                        if i%10 == 0:
                            torch.save(env.img[-1].detach().clone(), os.path.join(output, f'bundle_density_ep{i}.pt'))
                            torch.save(target_net_state_dict, os.path.join(output, f'model_state_dict_{name}.pt'))
                            torch.save(episode_durations, os.path.join(output, 'episode_durations_{name}.pt'))
                            torch.save(episode_returns, os.path.join(output, 'episode_returns_{name}.pt'))
                    break

                # if not terminated, move to the next state
                state = env.get_state()[0].to(dtype=torch.float32, device=DEVICE) # the head of the next streamline

        print('Complete')
        plot_durations(episode_durations, show_result=True)
        plot_returns(episode_durations, show_result=True)
        plt.ioff()
        plt.show()

        return