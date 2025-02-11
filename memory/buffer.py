
import random
import torch

from memory.tree import SumTree

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer():
    """
    A buffer to store and sample transitions for reinforcement learning.
    
    Parameters
    ----------
    capacity : int
        The maximum number of transitions that the buffer can hold.
    obs_shape : tuple
        The shape of the observation space.
    action_shape : tuple
        The shape of the action space.
        
    Attributes
    ----------
    obs : torch.Tensor
        Tensor to store observations.
    actions : torch.Tensor
        Tensor to store actions.
    next_obs : torch.Tensor
        Tensor to store next observations.
    rewards : torch.Tensor
        Tensor to store rewards.
    dones : torch.Tensor
        Tensor to store done flags.
    idx : int
        The current index for storing the next transition.
    full : bool
        Flag indicating if the buffer is full.
    capacity : int
        The maximum number of transitions that the buffer can hold.
    """
    
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
    

class PrioritizedReplayBuffer:
    """
    A buffer for storing and sampling transitions with prioritized experience replay.
    Reference: https://github.com/Howuhh/prioritized_experience_replay.git
    
    Parameters
    ----------
    capacity : int
        The maximum number of transitions that the buffer can hold.
    obs_shape : tuple
        The shape of the observation space.
    action_shape : tuple
        The shape of the action space.
    eps : float, optional
        A small positive constant to prevent zero priority, by default 1e-2.
    alpha : float, optional
        The exponent used in prioritization, by default 0.1.
    beta : float, optional
        The exponent used in importance sampling weights, by default 0.1.
        
    Attributes
    ----------
    eps : float
        A small positive constant to prevent zero priority.
    alpha : float
        The exponent used in prioritization.
    beta : float
        The exponent used in importance sampling weights.
    max_priority : float
        The maximum priority in the buffer.
    tree : SumTree
        A sum tree data structure for efficient sampling and updating of priorities.
    obs : torch.Tensor
        A tensor to store observations.
    actions : torch.Tensor
        A tensor to store actions.
    next_obs : torch.Tensor
        A tensor to store next observations.
    rewards : torch.Tensor
        A tensor to store rewards.
    dones : torch.Tensor
        A tensor to store done flags.
    idx : int
        The current index for inserting new transitions.
    full : bool
        A flag indicating whether the buffer is full.
    capacity : int
        The maximum number of transitions that the buffer can hold.
        
    """
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
        """
        Add a new experience to the buffer.
        Parameters
        ----------
        obs : object
            The current observation.
        action : object
            The action taken.
        next_obs : object
            The next observation after taking the action.
        reward : float
            The reward received after taking the action.
        done : bool
            Whether the episode has ended.
        """

        self.tree.add(self.max_priority, self.idx)

        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.next_obs[self.idx] = next_obs
        self.rewards[self.idx] = reward
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.idx == 0 or self.full
        

    def sample(self, batch_size):
        """
        Samples a batch of transitions from the buffer.
        
        Parameters
        ----------
        batch_size : int
            The number of transitions to sample.
            
        Returns
        -------
        obs : torch.Tensor
            The observations of the sampled transitions.
        actions : torch.Tensor
            The actions of the sampled transitions.
        next_obs : torch.Tensor
            The next observations of the sampled transitions.
        rewards : torch.Tensor
            The rewards of the sampled transitions.
        dones : torch.Tensor
            The done flags of the sampled transitions.
        weights : torch.Tensor
            The importance sampling weights of the sampled transitions.
        tree_idxs : list of int
            The indices of the sampled transitions in the priority tree.
            
        Raises
        ------
        AssertionError
            If the buffer contains fewer samples than the requested batch size.
        """
        
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
        """
        Update the priorities of the given data indices.
        
        Parameters
        ----------
        data_idxs : array-like
            Indices of the data whose priorities are to be updated.
        priorities : array-like or torch.Tensor
            New priorities for the data indices. If a torch.Tensor is provided, it will be converted to a numpy array.

        Notes
        -----
        The priorities are updated using the formula p_i = (|q_i| + eps) ** alpha, where eps is a small positive constant
        to prevent edge cases where priority is zero, and alpha is a scaling factor. The maximum priority is also updated accordingly.
        """
        
        if isinstance(priorities, torch.Tensor):
            priorities = priorities.detach().cpu().numpy()

        for data_idx, priority in zip(data_idxs, priorities):
            priority = priority.item()
            priority = (priority + self.eps) ** self.alpha
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    
    def __len__(self):
        return self.capacity if self.full else self.idx
    
if __name__ == "__main__":
    pass