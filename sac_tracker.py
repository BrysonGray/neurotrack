#!/usr/bin/env python

"""
Soft actor-critic tractography model functions

References
-----------
Reinforcement Learning: an Introduction, Sutton and Barto;
https://spinningup.openai.com/en/latest/algorithms/sac.html

Author: Bryson Gray
2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import os
import matplotlib
import matplotlib.pyplot as plt
from itertools import count
from tqdm import tqdm
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

    
class ReplayMemory():
    def __init__(self, capacity, obs_shape, action_shape):

        self.obs = torch.empty((capacity, *obs_shape), dtype=torch.float32, device='cpu')
        self.last_steps = torch.empty((capacity, 3), dtype=torch.float32, device='cpu')
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32, device='cpu')
        self.next_obs = torch.empty((capacity, *obs_shape), dtype=torch.float32, device='cpu')
        self.next_steps = torch.empty((capacity, 3), device='cpu')
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device='cpu')
        self.dones = torch.empty((capacity, 1), dtype=torch.bool, device='cpu')

        self.idx = 0
        self.full = False
        self.capacity = capacity
    
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
    
    def sample(self, batch_size, replacement=True):
        sample_range = self.capacity if self.full else self.idx
        if replacement:
            idxs = torch.randint(sample_range, size=(batch_size,))
        else:
            perm = torch.randperm(sample_range)
            idxs = perm[:batch_size]

        obs = self.obs[idxs].to(device=DEVICE)
        last_steps = self.last_steps[idxs].to(device=DEVICE)
        actions = self.actions[idxs].to(device=DEVICE)
        next_obs = self.next_obs[idxs].to(device=DEVICE)
        next_steps = self.next_steps[idxs].to(device=DEVICE)
        rewards = self.rewards[idxs].to(device=DEVICE)
        dones = self.dones[idxs].to(device=DEVICE)

        return obs, last_steps, actions, next_obs, next_steps, rewards, dones
    
    def __len__(self):
        return self.capacity if self.full else self.idx


class Critic(nn.Module):
    """
    Double Deep Q-Network CNN critic network model. This takes a state and an action as input,
    where the action is a direction in R^3 and an integer choice in {0,1,2} which represent
    step, terminate, and branch, respectively, and outputs a value.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    input_size : int
        Input image size of one dimension. Used to
        calculate the number of input features in the fully connected layer.
        Note the input must have equal height, width, and depth.
    n : int
        Channel size multiplier.
    """

    def __init__(self, in_channels, input_size, n=1):
        super().__init__()

        # calculate the size of the convolution output for input to Linear
        shape_out = lambda x, k, s: ((x - k)/s + 1)//1
        h = shape_out(input_size, 3, 2) # first conv
        h = shape_out(h, 3, 2) # second conv
        n_features = int(32*n * h**3)

        # convolution layers
        # 3 channels added to in_channels for components of previous step direction and 3 more for the action (chosen direction)
        self.conv1 = nn.Conv3d(in_channels + 6, 16*n, 3, stride=2)
        self.norm1 = nn.BatchNorm3d(16*n)
        self.conv2 = nn.Conv3d(16*n, 32*n, 3, stride=2)
        self.norm2 = nn.BatchNorm3d(32*n)

        # fully connected layers
        self.fc1 = nn.Linear(n_features, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, obs, last_step, direction):
        """
        Parameters
        ----------
        obs : torch.Tensor
            Observtion of the image patch with shape (M, C, L, L, L), where L is the side length of the patch.
        last_step : torch.Tensor
            Last step direction with shape (M, 3)
        action : torch.Tensor
            The action taken. This has shape (M, 4) where the first 3 components of each
            action is the step direction and the last component is the choice (step, terminate, branch).

        """
        # concatenate previous step direction components
        x = torch.concatenate((obs, torch.ones((obs.shape[0], 1, obs.shape[2], obs.shape[3], obs.shape[4]), device=DEVICE)*last_step[:,:,None,None,None]), dim=1)
        # concatenate new step direction
        x = torch.concatenate((x, torch.ones((x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4]), device=DEVICE)*direction[:,:,None,None,None]), dim=1)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    

class Actor(nn.Module):
    """
    Policy network model. This takes a state observation as input (image, last step), and outputs two distributions:
    one noraml distribution for step direction, and a categorical distribution for choice of
    step, branch, and terminate.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    input_size : int
        Input image size of one dimension. Used to
        calculate the number of input features in the fully connected layer.
        Note the input must have equal height, width, and depth.
    n : int
        Channel size multiplier.
    """
    def __init__(self, in_channels, input_size, n=1):
        super().__init__()

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
        obs = obs.to(device=DEVICE)
        step = step.to(device=DEVICE)
        x = torch.concatenate((obs, torch.ones((obs.shape[0], 1, obs.shape[2], obs.shape[3], obs.shape[4]), device=DEVICE)*step[:,:,None,None,None]), dim=1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x)
        # x = F.relu(self.norm1(self.conv1(x)))
        # x = F.relu(self.norm2(self.conv2(x)))
        x = self.conv2(x)
        x = self.norm2(x)
        x = F.relu(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        mu, std2, choice = torch.split(x, [3,3,3], dim=1) # the step direction will be mu / ||mu||
        # std2 = torch.sigmoid(std)
        std2 = torch.exp(std2)
        cov = torch.stack([torch.diagflat(s) for s in std2])

        direction_dist = torch.distributions.MultivariateNormal(mu, cov)
        choice_dist = torch.nn.functional.softmax(choice, dim=1)

        return direction_dist, choice_dist


class SACModel():
    def __init__(self, in_channels, input_size, start_steps=10000, lr=0.001, gamma=0.99, entropy_coeff=0.2):
    
        self.actor = Actor(in_channels, input_size).to(DEVICE)
        self.Q1 = Critic(in_channels, input_size).to(DEVICE)
        self.Q2 = Critic(in_channels, input_size).to(DEVICE)
        self.Q1_target = Critic(in_channels, input_size).to(DEVICE)
        self.Q2_target = Critic(in_channels, input_size).to(DEVICE)
        # initialize the Q1 and Q2 target networks with the same params as Q1 and Q2
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

        self.Q1_optimizer = optim.AdamW(self.Q1.parameters(), lr=lr, amsgrad=True) # type: ignore
        self.Q2_optimizer = optim.AdamW(self.Q2.parameters(), lr=lr, amsgrad=True) # type: ignore
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr, amsgrad=True) # type: ignore

        self.criterion = nn.MSELoss()

        # self.lrscheduler = torch.optim.lr_scheduler.StepLR(self.Q1_optimizer, step_size=1, gamma=0.9993)

        self.memory = ReplayMemory(10000, obs_shape=(in_channels,input_size,input_size,input_size), action_shape=(4,))

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
        batch_size = obs.shape[0]
        if steps_done < self.start_steps:
            direction = torch.randn((batch_size, 3))
            direction = direction / (torch.linalg.norm(direction, dim=1, keepdims=True) + torch.finfo(torch.float).eps)
            # Do not use a uniform distribution over choices because our prior expectation is it
            # will step much more often than branching or terminating a path. It should also
            # terminate slightly more than branch otherwise an episode could run indefinitely.
            choice_dist = torch.tensor([0.98, 0.012, 0.008])[None] * torch.ones((batch_size, 1))
            choice = torch.multinomial(choice_dist, num_samples=1) # (batch_size, 1)
        else:
            direction_dist, choice_dist = self.actor(obs, last_step) 
            if sample:
                direction = direction_dist.rsample()
                direction = direction / (torch.linalg.norm(direction, dim=1)[:,None] + torch.finfo(torch.float).eps)
                choice = torch.multinomial(choice_dist, num_samples=batch_size, replacement=True) # (batch_size, 1)
            else:
                direction = direction_dist.mean
                direction = direction / (torch.linalg.norm(direction, dim=1)[:,None] + torch.finfo(torch.float).eps)
                choice = torch.argmax(choice_dist, dim=1)[:,None] # (batch_size, 1)

        action = torch.cat((direction, choice), dim=-1)

        return action.to(device=DEVICE)


    def optimize_Q(self, obs, last_steps, actions, next_obs, next_steps, rewards, dones):
        """ Do one Q function optimization step
        
        """
        # compute targets
        # sample next actions from the current policy
        next_action = self.select_action(next_obs, next_steps)
        next_direction = next_action[...,:3]
        next_choice = next_action[...,3].to(dtype=torch.int)

        # get log-probs of next actions
        direction_dist, choice_dist = self.actor(next_obs, next_steps)
        log_prob = direction_dist.log_prob(next_direction) + torch.log(choice_dist[range(len(next_choice)),next_choice]) #choice_dist.log_prob(next_choice)
        log_prob = log_prob.detach()

        # get target q-values
        Q1_target_vals_vec = self.Q1_target(next_obs, next_steps, next_direction) # vector of q-values for each choice
        Q2_target_vals_vec = self.Q2_target(next_obs, next_steps, next_direction)
        Q1_target_vals = Q1_target_vals_vec[range(len(next_choice)),next_choice]
        Q2_target_vals = Q2_target_vals_vec[range(len(next_choice)),next_choice]
        targets = rewards.squeeze() + self.gamma * torch.logical_not(dones.squeeze()) * (torch.minimum(Q1_target_vals, Q2_target_vals) - self.entropy_coeff * log_prob)
        targets = targets.detach()

        # compute q-values to compare against targets
        Q1_vals_vec = self.Q1(obs, last_steps, actions[:,:3])
        Q1_vals = Q1_vals_vec[range(len(actions)), actions[:,3].to(dtype=torch.int)]
        Q1_loss = self.criterion(Q1_vals, targets)
        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), 1)
        self.Q1_optimizer.step()

        Q2_vals_vec = self.Q2(obs, last_steps, actions[:,:3])
        Q2_vals = Q2_vals_vec[range(len(actions)), actions[:,3].to(dtype=torch.int)]
        Q2_loss = self.criterion(Q2_vals, targets)
        self.Q2_optimizer.zero_grad()
        Q2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), 1)
        self.Q2_optimizer.step()        

        return
    

    def optimize_policy(self, obs, last_steps):
        """ Do one policy function optimization step. """
        # sample actions from policy
        action = self.select_action(obs, last_steps)
        direction = action[...,:3]
        choice = action[...,3].to(dtype=torch.int)

        # get log-probs of actions
        direction_dist, choice_dist = self.actor(obs, last_steps)
        log_prob = direction_dist.log_prob(direction) + torch.log(choice_dist[range(len(choice)),choice]) # choice_dist.log_prob(choice)

        # get expected Q-vals
        Q1_vals_vec = self.Q1(obs, last_steps, direction)
        Q2_vals_vec = self.Q1(obs, last_steps, direction)
        expected_Q1 = torch.einsum('ij,ij->i', Q1_vals_vec, choice_dist)
        expected_Q2 = torch.einsum('ij,ij->i', Q2_vals_vec, choice_dist)
        Q_min = torch.minimum(expected_Q1,expected_Q2)

        # entropy regularized Q values
        loss = (self.entropy_coeff * log_prob - Q_min).mean() # The loss function is multiplied by -1 to do gradient ascent instead of decent.

        self.actor_optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)

        self.actor_optimizer.step()

        return loss.cpu()
    

    def train(self,
              env,
              episodes=100,
              batch_size=100,
              update_after=1000,
              update_every=50,
              tau=0.995,
              save_snapshots=True,
              output='./outputs',
              name='config0',
              show=False):

        steps_done = 0
        episode_durations = []
        episode_returns = []
        losses = []
        grad_norms = []
        # lr_vals = []
        mae = []
        bending_energy = []
        friction = []
        n_paths = []

        if not os.path.exists(os.path.join(output, name)):
            os.makedirs(os.path.join(output, name))
        output = os.path.join(output, name)

        # Train the Network
        for i in tqdm(range(episodes)):
            env.reset()
            obs, last_step = env.get_state()
            ep_return = 0
            for t in count():
                action = self.select_action(obs, last_step, steps_done, sample=True)[0] # shape (4)
                steps_done += 1
                # take step, get observation and reward, and move index to next streamline
                observation, reward, terminated = env.step(action)
                ep_return += reward.detach().cpu()

                next_obs, next_step = observation
                # Store the transition in memory
                self.memory.push(obs.detach().cpu(), last_step.detach().cpu(), action.detach().cpu(), next_obs.detach().cpu(), next_step.detach().cpu(), reward.detach().cpu(), terminated)
                
                # Perform updates once there is sufficient transitions saved.
                if steps_done >= update_after:
                    if steps_done % update_every == 0:
                        obs, last_steps, actions, next_obs, next_steps, rewards, dones = self.memory.sample(batch_size, replacement=True)

                        # Perform one step of the optimization on the Q networks.
                        self.optimize_Q(obs, last_steps, actions, next_obs, next_steps, rewards, dones)
                        # Soft update of the target network's weights
                        # θ′ ← τ θ + (1 −τ )θ′
                        for Q,Q_target in zip([self.Q1, self.Q2], [self.Q1_target, self.Q2_target]):
                            Q_state_dict = Q.state_dict()
                            Q_target_state_dict = Q_target.state_dict()
                            for key in Q_state_dict:
                                Q_target_state_dict[key] = Q_state_dict[key]*tau + Q_target_state_dict[key]*(1-tau)
                            Q_target.load_state_dict(Q_target_state_dict)

                        # Perform one step of optimization on the policy network
                        loss = self.optimize_policy(obs, last_steps)

                        # record policy function loss
                        if loss:
                            losses.append(loss)
                            total_norm = 0
                            for p in self.actor.parameters():
                                param_norm = p.grad.norm(2) # type: ignore
                                total_norm = total_norm + param_norm.detach().item() ** 2
                            total_norm = total_norm ** (1. / 2)
                            grad_norms.append(total_norm)

                if terminated:
                    episode_durations.append(t + 1)
                    episode_returns.append(ep_return)
                    n_paths.append(len(env.finished_paths))
                    # save global matching error
                    # mae.append(torch.mean(torch.abs(env.bundle_density.data - env.true_density.data)))
                    mae_ = torch.mean(torch.abs(env.img.data[3] - env.true_density.data)).cpu()
                    mae.append(mae_)
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
                            torch.save(self.actor.state_dict(), os.path.join(output, f'policy_state_dict_{name}.pt'))
                            torch.save(self.Q1_target.state_dict(), os.path.join(output, f'Q1_state_dict_{name}.pt'))
                            torch.save(self.Q2_target.state_dict(), os.path.join(output, f'Q2_state_dict_{name}.pt'))
                            torch.save(episode_durations, os.path.join(output, f'episode_durations_{name}.pt'))
                            torch.save(episode_returns, os.path.join(output, f'episode_returns_{name}.pt'))
                            torch.save(mae, os.path.join(output, f'matching_error_{name}.pt'))
                            torch.save(bending_energy, os.path.join(output, f'bending_energy_{name}.pt'))
                            torch.save(friction, os.path.join(output, f'friction_{name}.pt'))
                            torch.save(n_paths, os.path.join(output, f'n_paths_{name}.pt'))

                            torch.save(losses, os.path.join(output, f'loss_{name}.pt'))
                            torch.save(grad_norms, os.path.join(output, f'grad_norms_{name}.pt'))
                            # torch.save(lr_vals, os.path.join(output, f'lr_{name}.pt'))
                    break

                # if not terminated, move to the next state
                obs, last_step = env.get_state() # the head of the next streamline
                
            # if len(self.memory) > batch_size:
            #     self.lrscheduler.step()
            #     lr = self.lrscheduler.get_last_lr()
            #     lr_vals.append(lr)

        print('Complete')
        if show:
            plt.ioff()
            plt.show()

        return
    

    def inference(self, env, out=None):

            env.reset()

            obs, last_step = env.get_state()
            obs = obs.to(dtype=torch.float32, device=DEVICE)
            last_step = last_step.to(dtype=torch.float32, device=DEVICE)

            ep_return = 0
            i = 0
            plt.ioff()
            
            while True:
                # get action
                action = self.select_action(obs, last_step, sample=False)

                # take step, get observation and reward, and move index to next streamline
                observation, reward, terminated = env.step(action) 
                ep_return += reward
                i += 1

                if terminated:
                    return env

                else:
                    # if not terminated, move to the next state
                    obs, last_step = env.get_state()
                    obs = obs.to(dtype=torch.float32, device=DEVICE)
                    last_step = last_step.to(dtype=torch.float32, device=DEVICE)
                    if out is not None and i%10 == 0:
                        plt.figure(0)
                        plt.imshow(env.img.data[-1].amax(dim=0), cmap='hot', alpha=0.5)#, int(paths[env.head_id][-1, 0])])
                        plt.imshow(env.img.data[:3].amin(dim=1).permute(1,2,0), alpha=0.5)
                        plt.axis('off')
                        plt.savefig(os.path.join(out, f'path_{i}.png'))