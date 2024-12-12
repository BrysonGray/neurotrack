
import random
import torch

from memory.tree import SumTree

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
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
    

"""
Reference: https://github.com/Howuhh/prioritized_experience_replay.git
"""
class PrioritizedReplayBuffer:
    def __init__(self, capacity, obs_shape, action_shape, eps=1e-2, alpha=0.1, beta=0.1):
        self.eps = eps
        self.alpha = alpha
        self.beta = beta
        self.max_priority = eps
        self.tree = SumTree(size=capacity)

        self.obs = torch.empty((capacity, *obs_shape), dtype=torch.float32, device='cpu')
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32, device='cpu')
        self.next_obs = torch.empty((capacity, *obs_shape), dtype=torch.float32, device='cpu')
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device='cpu')
        self.dones = torch.empty((capacity, 1), dtype=torch.bool, device='cpu')

        self.idx = 0
        self.full = False
        self.capacity = capacity
    
    def push(self, obs, action, next_obs, reward, done):

        self.tree.add(self.max_priority, self.idx)

        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.next_obs[self.idx] = next_obs
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.idx == 0 or self.full

    def sample(self, batch_size):
        real_size = len(self)
        assert real_size >= batch_size, "buffer contains less samples than batch size"
        
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # sample_idx is a sample index in buffer, needed further to sample actual transitions
            # tree_idx is a index of a sample in the tree, needed further to update priorities
            tree_idx, priority, sample_idx = self.tree.get(cumsum)

            priorities[i] = priority
            tree_idxs.append(tree_idx)
            sample_idxs.append(sample_idx)
        
        probs = priorities / self.tree.total
        weights = (real_size * probs) ** -self.beta

        weights = weights / weights.max()

        obs = self.obs[sample_idxs].to(DEVICE)
        actions = self.actions[sample_idxs].to(DEVICE)
        next_obs = self.next_obs[sample_idxs].to(DEVICE)
        rewards = self.rewards[sample_idxs].to(DEVICE)
        dones = self.dones[sample_idxs].to(DEVICE)

        return obs, actions, next_obs, rewards, dones, weights, tree_idxs
    
    def update_priorities(self, data_idxs, priorities):
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            # The first variant we consider is the direct, proportional prioritization where p_i = |δ_i| + eps,
            # where eps is a small positive constant that prevents the edge-case of transitions not being
            # revisited once their error is zero. (Section 3.3)
            priority = priority.item()
            priority = (priority + self.eps) ** self.alpha

            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    
    def __len__(self):
        return self.capacity if self.full else self.idx