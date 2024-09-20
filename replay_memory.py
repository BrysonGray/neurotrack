
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory():
    def __init__(self, capacity, obs_shape, action_shape):

        self.obs = torch.empty((capacity, *obs_shape), dtype=torch.float32, device='cpu')
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32, device='cpu')
        self.next_obs = torch.empty((capacity, *obs_shape), dtype=torch.float32, device='cpu')
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device='cpu')
        self.dones = torch.empty((capacity, 1), dtype=torch.bool, device='cpu')

        self.idx = 0
        self.full = False
        self.capacity = capacity
    
    def push(self, obs, action, next_obs, reward, done):
        """Save a transition to replay memory"""
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.next_obs[self.idx] = next_obs
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.idx == 0 or self.full
    
    def sample(self, batch_size, replacement=False):
        sample_range = self.capacity if self.full else self.idx
        if replacement:
            idxs = torch.randint(sample_range, size=(batch_size,))
        else:
            perm = torch.randperm(sample_range)
            idxs = perm[:batch_size]

        obs = self.obs[idxs].to(device=DEVICE)
        actions = self.actions[idxs].to(device=DEVICE)
        next_obs = self.next_obs[idxs].to(device=DEVICE)
        rewards = self.rewards[idxs].to(device=DEVICE)
        dones = self.dones[idxs].to(device=DEVICE)

        return obs, actions, next_obs, rewards, dones
    
    def __len__(self):
        return self.capacity if self.full else self.idx
