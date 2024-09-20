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

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter

import os
from itertools import count
from tqdm import tqdm

from replay_memory import ReplayMemory

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def shape_out(input_size, kernel_size, stride):
    return ((input_size - kernel_size)/stride + 1)//1


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

    def __init__(self, in_channels, input_size, n_filters=16):
        super().__init__()
        
        # calculate the size of the convolution output for input to Linear
        c1, k1, s1 = (n_filters, 3, 2)
        c2, k2, s2 = (n_filters*2, 3, 2)
        h = shape_out(input_size, k1, s1) # first conv
        h = shape_out(h, k2, s2) # second conv
        n_features = int(c2 * h**3)

        # convolution layers
        # 3 channels added to in_channels for components of the action (chosen direction)
        self.conv1 = nn.Conv3d(in_channels + 3, c1, k1, stride=s1)
        self.norm1 = nn.BatchNorm3d(c1)
        self.conv2 = nn.Conv3d(c1, c2, k2, stride=s2)
        self.norm2 = nn.BatchNorm3d(c2)

        # fully connected layers
        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, obs, direction):
        """
        Parameters
        ----------
        obs : torch.Tensor
            Observtion of the image patch with shape (M, C, L, L, L), where L is the side length of the patch.
        last_step : torch.Tensor
            Last step direction with shape (M, 3)
        action : torch.Tensor
            The action taken. This has shape (M, 3) where each action is 3 step direction components.

        """

        # concatenate new step direction
        x = torch.concatenate((obs, torch.ones((obs.shape[0], 1, obs.shape[2], obs.shape[3], obs.shape[4]), device=DEVICE)*direction[:,:,None,None,None]), dim=1)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
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
    def __init__(self, in_channels, input_size, n_filters=16):
        super().__init__()

        # calculate the size of the convolution output for input to Linear
        c1, k1, s1 = (n_filters, 3, 2)
        c2, k2, s2 = (n_filters*2, 3, 2)
        h = shape_out(input_size, k1, s1) # first conv
        h = shape_out(h, k2, s2) # second conv
        n_features = int(c2 * h**3)

        # convolution layers
        self.conv1 = nn.Conv3d(in_channels, c1, k1, stride=s1)
        self.norm1 = nn.BatchNorm3d(c1)
        self.conv2 = nn.Conv3d(c1, c2, k2, stride=s2)
        self.norm2 = nn.BatchNorm3d(c2)

        # fully connected layers
        self.fc1 = nn.Linear(n_features, 256)
        self.fc2 = nn.Linear(256, 256)
        # The last fully connected layer outputs three means and standard deviations for the step direction components,
        # plus the three element logits representing the choice between taking a step, terminating the path, or branching.
        self.fc3 = nn.Linear(256, 4)
    
    def forward(self, obs):
        obs = obs.to(device=DEVICE)
        x = self.conv1(obs)
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
        
        mu, logvar = torch.split(x, [3,1], dim=1)
        mu = torch.sigmoid(mu)*10 # put a boundary on the length of a step.
        logvar = torch.sigmoid(logvar) * 3 + 1
        var = torch.exp(logvar*0.5)
        # cov = torch.stack([torch.diagflat(s) for s in var])

        # direction_dist = torch.distributions.MultivariateNormal(mu, cov)

        direction_dist = (torch.randn(x.shape[0],3).to(device=DEVICE))*var + mu
        direction_dist = direction_dist * torch.tensor([[0, 1, 1]], device=DEVICE)

        return direction_dist


class SACModel():
    def __init__(self,
                 in_channels,
                 input_size,
                 start_steps=10000,
                 lr=0.001,
                 gamma=0.99,
                 entropy_coeff=0.2,
                 n_filters=16):
    
        self.actor = Actor(in_channels, input_size, n_filters).to(DEVICE)
        self.Q1 = Critic(in_channels, input_size, n_filters).to(DEVICE)
        # self.Q2 = Critic(in_channels, input_size, n_filters).to(DEVICE)
        self.Q1_target = Critic(in_channels, input_size, n_filters).to(DEVICE)
        # self.Q2_target = Critic(in_channels, input_size, n_filters).to(DEVICE)
        # initialize the Q1 and Q2 target networks with the same params as Q1 and Q2
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        # self.Q2_target.load_state_dict(self.Q2.state_dict())

        self.Q1_optimizer = optim.AdamW(self.Q1.parameters(), lr=lr, amsgrad=True) # type: ignore
        # self.Q2_optimizer = optim.AdamW(self.Q2.parameters(), lr=lr, amsgrad=True) # type: ignore
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr, amsgrad=True) # type: ignore

        self.criterion = nn.MSELoss()

        # self.lrscheduler = torch.optim.lr_scheduler.StepLR(self.Q1_optimizer, step_size=1, gamma=0.9993)

        self.memory = ReplayMemory(10000, obs_shape=(in_channels,input_size,input_size,input_size), action_shape=(3,))

        self.start_steps = start_steps # random steps taken before beginning to sample from the policy
        self.gamma = gamma # discount factor
        self.entropy_coeff = entropy_coeff


    def load_model(self, actor_weights, Q1_weights, Q2_weights):
        self.Q1.load_state_dict(Q1_weights)
        # self.Q2.load_state_dict(Q1_weights)
        self.Q1_target.load_state_dict(Q2_weights)
        # self.Q2_target.load_state_dict(Q2_weights)
        self.actor.load_state_dict(actor_weights)


    def select_action(self, obs, steps_done=0, sample=True):
        batch_size = obs.shape[0]
        if steps_done < self.start_steps:
            direction = torch.randn((batch_size, 3))

        else:
            direction = self.actor(obs) 
            # if sample:
            #     direction = direction_dist.rsample()

            # else:
            #     direction = direction_dist.mean

        return direction.to(device=DEVICE)


    def optimize_Q(self, obs, actions, next_obs, rewards, dones):
        """ Do one Q function optimization step
        
        """
        # compute targets
        with torch.no_grad():
            # sample next actions from the current policy
            next_direction = self.select_action(next_obs, steps_done=self.start_steps) # set steps_done to start_steps so that this samples from the current policy

            # get log-probs of next actions
            direction_dist = self.actor(next_obs)
            log_prob = 0.0 #direction_dist.log_prob(next_direction) 
            # log_prob = log_prob.detach()

            # get target q-values
            Q1_target_vals = self.Q1_target(next_obs, next_direction) # vector of q-values for each choice
            # Q2_target_vals = self.Q2_target(next_obs, next_direction).squeeze()
            # targets = rewards.squeeze() + self.gamma * torch.logical_not(dones.squeeze()) * (torch.minimum(Q1_target_vals, Q2_target_vals) - self.entropy_coeff * log_prob)
            targets = rewards + self.gamma * torch.logical_not(dones) * (Q1_target_vals - self.entropy_coeff * log_prob)

        # compute q-values to compare against targets
        Q1_vals = self.Q1(obs, actions)
        Q1_loss = self.criterion(Q1_vals, targets)
        # print("Q1_loss:", Q1_loss.detach())
        # print("reward:", rewards.squeeze())
        self.Q1_optimizer.zero_grad()
        Q1_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.Q1.parameters(), 1)
        self.Q1_optimizer.step()

        # Q2_vals = self.Q2(obs, actions)[:,0]
        # Q2_loss = self.criterion(Q2_vals, targets)
        # self.Q2_optimizer.zero_grad()
        # Q2_loss.backward()
        # # torch.nn.utils.clip_grad_norm_(self.Q2.parameters(), 1)
        # self.Q2_optimizer.step()

        return
    

    def optimize_policy(self, obs):
        """ Do one policy function optimization step. """
        # sample actions from policy
        direction = self.select_action(obs, steps_done=self.start_steps)

        # get log-probs of actions
        # direction_dist = self.actor(obs)
        # log_prob = direction_dist.log_prob(direction)
        log_prob = 0.0

        # get expected Q-vals
        Q1_vals = self.Q1(obs, direction)[:,0]
        # Q2_vals = self.Q2(obs, direction)[:,0]

        # Q_min = torch.minimum(Q1_vals,Q2_vals)
        Q_min = Q1_vals

        # entropy regularized Q values
        loss = (self.entropy_coeff * log_prob - Q_min).mean() # The loss function is multiplied by -1 to do gradient ascent instead of decent.

        self.actor_optimizer.zero_grad()
        loss.backward()

        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)

        self.actor_optimizer.step()

        return loss.cpu()
    

    def train(self,
              env,
              episodes=100,
              batch_size=100,
              update_after=1000,
              update_every=50,
              tau=0.995,
              output='./outputs',
              name='config0',
              show=False):

        steps_done = 0
        num_updates = 0
        gr_max = 0
        episode_returns = []


        if not os.path.exists(os.path.join(output, name)):
            os.makedirs(os.path.join(output, name))
        output = os.path.join(output, name)

        writer = SummaryWriter(output)

        # Train the Network
        for i in tqdm(range(episodes)):
            obs = env.get_state()
            ep_return = 0
            for t in count():
                action = self.select_action(obs, steps_done, sample=True)[0] # get 0th axis to remove batch dim
                steps_done += 1
                # take step, get observation and reward, and move index to next streamline
                next_obs, reward, terminated = env.step(action)
                ep_return += reward.detach().cpu()

                # Store the transition in memory
                self.memory.push(obs.detach().cpu(), action.detach().cpu(), next_obs.detach().cpu(), reward.detach().cpu(), terminated)
                
                # Perform updates once there is sufficient transitions saved.
                if steps_done >= update_after:
                    if steps_done == update_after:
                        print("Optimization begun")
                    if steps_done % update_every == 0:
                        policy_loss = 0
                        for i in range(update_every):
                            num_updates += 1
                            obs, actions, next_obs, rewards, dones = self.memory.sample(batch_size, replacement=False)

                            # Perform one step of the optimization on the Q networks.
                            self.optimize_Q(obs, actions, next_obs, rewards, dones)
                            # Soft update of the target network's weights
                            # θ′ ← τ θ + (1 −τ )θ′
                            # for Q,Q_target in zip([self.Q1, self.Q2], [self.Q1_target, self.Q2_target]):
                            #     Q_state_dict = Q.state_dict()
                            #     Q_target_state_dict = Q_target.state_dict()
                            #     for key in Q_state_dict:
                            #         Q_target_state_dict[key] = Q_state_dict[key]*tau + Q_target_state_dict[key]*(1-tau)
                            #     Q_target.load_state_dict(Q_target_state_dict)
                            
                            target_net_state_dict = self.Q1_target.state_dict()
                            policy_net_state_dict = self.Q1.state_dict()
                            for key in policy_net_state_dict:
                                target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
                            self.Q1_target.load_state_dict(target_net_state_dict)

                            # Perform one step of optimization on the policy network
                            loss = self.optimize_policy(obs)
                            policy_loss = policy_loss + loss

                        writer.add_scalar("policy loss",
                                          policy_loss/update_every,
                                          steps_done)


                if terminated:
                    # episode_durations.append(t + 1)
                    episode_returns.append(ep_return)
                    # n_paths.append(len(env.finished_paths))
                    # save global recall (TP / (TP + FN))
                    true_neuron = env.true_density.data > 0.0
                    labeled_neuron = env.img.data[3] > 0.0
                    TP = torch.logical_and(true_neuron, labeled_neuron)
                    tot = true_neuron.sum() # (TP + FN)
                    gr = torch.sum(TP/tot)
                    if gr > gr_max:
                        gr_max = gr
                    writer.add_scalar("num steps", t+1, steps_done)
                    writer.add_scalar("returns", ep_return, steps_done)
                    # writer.add_scalar("returns", returns_avg[i], steps_done)
                    writer.add_scalar("num paths", len(env.finished_paths), steps_done)
                    writer.add_scalar("global recall", gr, steps_done)

                    # save global bending energy
                    bending_energy_ = []
                    for j in range(len(env.finished_paths)):
                        p0 = env.finished_paths[j][:-1]
                        p1 = env.finished_paths[j][1:]
                        segments = (p1 - p0) #/ env.step_size
                        segments = segments / torch.linalg.norm(segments, dim=1, keepdims=True)
                        energy = (torch.einsum('ij,ij->i', segments[1:], segments[:-1]) - 1.0) / -2.0
                        bending_energy_.append(torch.sum(energy))
                    # bending_energy.append(torch.sum(torch.tensor(bending_energy_)))
                    writer.add_scalar("bending energy", torch.sum(torch.tensor(bending_energy_)), steps_done)
                    
                    # if show:
                    #     env_utils.plot_durations(episode_durations)
                    #     env_utils.plot_returns(episode_returns)

                    # if the average return increases, save the model dicts
                    # if gr == gr_max:
                    #     model_dicts = {"policy_state_dict": self.actor.state_dict(),
                    #                     "Q1_state_dict": self.Q1_target.state_dict(),
                    #                     "Q2_state_dict": self.Q2_target.state_dict(),}
                    #     torch.save(model_dicts, os.path.join(output, f"model_state_dicts_{name}.pt"))

                    if i % 10 == 0:
                        fig, ax = plt.subplots(1,3, figsize=[p*2 for p in plt.rcParams["figure.figsize"]])
                        path = env.img.data[3].cpu()
                        mask = env.mask.data[0].cpu()
                        true_density = env.true_density.data[0].cpu()
                        toshow = torch.stack([path,true_density,mask], dim=-1)
                        ax[0].imshow(toshow.amax(0))
                        ax[1].imshow(toshow.amax(1))
                        ax[2].imshow(toshow.amax(2))
                        fig.suptitle(f"Return: {ep_return.item():.4f}, Length: {t+1},\n\
                                       Recall: {gr:.4f}, N-Paths: {len(env.finished_paths)}")
                        writer.add_figure("Sample Path", fig, global_step=steps_done)      

                    env.reset()
                    break

                # if not terminated, move to the next state
                obs = env.get_state() # the head of the next streamline
                
            # if len(self.memory) > batch_size:
            #     self.lrscheduler.step()
            #     lr = self.lrscheduler.get_last_lr()
            #     lr_vals.append(lr)

        print('Complete')
        if show:
            plt.ioff()
            plt.show()

        return
    

    def inference(self, env, out=os.getcwd()):

            if not os.path.isdir(out):
                os.makedirs(out)
            env.reset()

            obs = env.get_state()
            obs = obs.to(dtype=torch.float32, device=DEVICE)

            ep_return = 0
            i = 0
            plt.ioff()
            
            while True:
                # get action
                action = self.select_action(obs, sample=False)[0]

                # take step, get observation and reward, and move index to next streamline
                observation, reward, terminated = env.step(action) 
                ep_return += reward
                i += 1

                if terminated:
                    return env

                else:
                    # if not terminated, move to the next state
                    obs = env.get_state()
                    obs = obs.to(dtype=torch.float32, device=DEVICE)
                    if out is not None and i%10 == 0:
                        plt.figure(0)
                        plt.imshow(env.img.data[-1].cpu().amax(dim=0), cmap='hot', alpha=0.5)#, int(paths[env.head_id][-1, 0])])
                        plt.imshow(env.img.data[:3].cpu().amin(dim=1).permute(1,2,0), alpha=0.5)
                        plt.axis('off')
                        plt.savefig(os.path.join(out, f'path_{i}.png'))