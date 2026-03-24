"""
This module implements a replay buffer and a prioritized replay buffer for storing and sampling.

Version 2 (v2) refactors the buffer to store target vectors rather than rewards so that the reward gradient can be calculated during the actor update step.
"""
import random
import torch
import numpy as np

from neurotrack.training.tree import SumTree

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _require_uint8_observation(observation, name):
    """Validate and normalize an observation tensor to contiguous CPU uint8."""
    if isinstance(observation, torch.Tensor):
        observation_t = observation.detach().to(device="cpu")
    else:
        observation_t = torch.as_tensor(observation, device="cpu")

    if observation_t.dtype != torch.uint8:
        raise TypeError(
            f"{name} must be torch.uint8 before model input conversion, got {observation_t.dtype}."
        )
    return observation_t.contiguous()


def _normalize_target_vectors(target_vectors):
    """Convert a target vector set to a contiguous CPU tensor of shape (K, 3)."""
    if isinstance(target_vectors, torch.Tensor):
        target_t = target_vectors.detach().to(device="cpu", dtype=torch.float32)
    else:
        target_t = torch.as_tensor(target_vectors, dtype=torch.float32, device="cpu")
    if target_t.ndim == 1:
        target_t = target_t.unsqueeze(0)
    if target_t.ndim != 2 or target_t.shape[1] != 3:
        raise ValueError(f"target_vectors must have shape (K, 3), got {tuple(target_t.shape)}")
    return target_t.contiguous()


def _normalize_bc_observation(observation):
    """Normalize BC observations to contiguous uint8 tensors of shape (C, D, H, W)."""
    obs_t = _require_uint8_observation(observation, name="obs")
    if obs_t.ndim == 5:
        if obs_t.shape[0] != 1:
            raise ValueError(
                f"BC observation with 5 dimensions must be batched as (1, C, D, H, W), got {tuple(obs_t.shape)}"
            )
        obs_t = obs_t[0]
    if obs_t.ndim != 4:
        raise ValueError(f"BC observations must have shape (C, D, H, W), got {tuple(obs_t.shape)}")
    return obs_t.contiguous()


def _pad_target_vector_sets(target_vector_sets, device):
    """Pad a batch of variable-length target vector sets and return a validity mask."""
    batch_size = len(target_vector_sets)
    if batch_size == 0:
        return (
            torch.empty((0, 0, 3), dtype=torch.float32, device=device),
            torch.empty((0, 0), dtype=torch.bool, device=device),
        )

    max_targets = max(target_vectors.shape[0] for target_vectors in target_vector_sets)
    padded = torch.zeros((batch_size, max_targets, 3), dtype=torch.float32, device=device)
    valid_mask = torch.zeros((batch_size, max_targets), dtype=torch.bool, device=device)

    for row, target_vectors in enumerate(target_vector_sets):
        target_vectors = target_vectors.to(device=device)
        count = target_vectors.shape[0]
        padded[row, :count] = target_vectors
        valid_mask[row, :count] = True

    return padded, valid_mask


def _permute_vector_components(vectors, component_indices):
    """Permute the coordinate dimension of action or target-vector tensors."""
    index = torch.tensor(component_indices, dtype=torch.long, device=vectors.device)
    return vectors.index_select(-1, index)


def _transform_batch(
    obs,
    actions,
    next_obs,
    current_target_vectors,
    current_target_mask,
    next_target_vectors,
    next_target_mask,
    include_z_flip,
):
    """Apply the same spatial augmentation to observations, actions, and target vectors."""
    perm = torch.randperm(2) + 3
    obs = obs.permute([0, 1, 2, *perm])
    next_obs = next_obs.permute([0, 1, 2, *perm])
    i, j = [x.item() - 2 for x in perm]
    component_indices = [0, i, j]
    actions = _permute_vector_components(actions, component_indices)
    current_target_vectors = _permute_vector_components(current_target_vectors, component_indices)
    next_target_vectors = _permute_vector_components(next_target_vectors, component_indices)

    if torch.rand(1) > 0.5:
        obs = obs.flip(-1)
        next_obs = next_obs.flip(-1)
        actions[..., -1] = -actions[..., -1]
        current_target_vectors[..., -1] = -current_target_vectors[..., -1]
        next_target_vectors[..., -1] = -next_target_vectors[..., -1]

    if torch.rand(1) > 0.5:
        obs = obs.flip(-2)
        next_obs = next_obs.flip(-2)
        actions[..., -2] = -actions[..., -2]
        current_target_vectors[..., -2] = -current_target_vectors[..., -2]
        next_target_vectors[..., -2] = -next_target_vectors[..., -2]

    if include_z_flip and torch.rand(1) > 0.5:
        obs = obs.flip(-3)
        next_obs = next_obs.flip(-3)
        actions[..., -3] = -actions[..., -3]
        current_target_vectors[..., -3] = -current_target_vectors[..., -3]
        next_target_vectors[..., -3] = -next_target_vectors[..., -3]

    return (
        obs,
        actions,
        next_obs,
        current_target_vectors,
        current_target_mask,
        next_target_vectors,
        next_target_mask,
    )


def _transform_bc_batch(obs, target_vectors, target_mask, include_z_flip):
    """Apply SAC-style spatial augmentation to BC observations and target vectors."""
    perm = torch.randperm(2) + 3
    obs = obs.permute([0, 1, 2, *perm])
    i, j = [x.item() - 2 for x in perm]
    component_indices = [0, i, j]
    target_vectors = _permute_vector_components(target_vectors, component_indices)

    if torch.rand(1) > 0.5:
        obs = obs.flip(-1)
        target_vectors[..., -1] = -target_vectors[..., -1]

    if torch.rand(1) > 0.5:
        obs = obs.flip(-2)
        target_vectors[..., -2] = -target_vectors[..., -2]

    if include_z_flip and torch.rand(1) > 0.5:
        obs = obs.flip(-3)
        target_vectors[..., -3] = -target_vectors[..., -3]

    return obs, target_vectors, target_mask


class BehaviorCloningReplayBuffer:
    """FIFO replay buffer for behavior-cloning supervision with optional augmentation on sampling."""

    def __init__(self, capacity, include_z_flip=True):
        if int(capacity) < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = int(capacity)
        self.include_z_flip = bool(include_z_flip)
        self.obs = None
        self.target_vectors = [None] * self.capacity
        self.stop_labels = torch.empty((self.capacity,), dtype=torch.bool, device="cpu")
        self.idx = 0
        self.full = False

    def _ensure_obs_storage(self, obs_shape):
        if self.obs is None:
            self.obs = torch.empty((self.capacity, *obs_shape), dtype=torch.uint8, device="cpu")
            return
        if tuple(self.obs.shape[1:]) != tuple(obs_shape):
            raise ValueError(
                f"All observations in BehaviorCloningReplayBuffer must share one shape. "
                f"Expected {tuple(self.obs.shape[1:])}, got {tuple(obs_shape)}."
            )

    def push(self, obs, target_vectors, stop_label=False):
        obs_t = _normalize_bc_observation(obs)
        self._ensure_obs_storage(tuple(obs_t.shape))
        self.obs[self.idx] = obs_t
        self.target_vectors[self.idx] = _normalize_target_vectors(target_vectors)
        self.stop_labels[self.idx] = bool(stop_label)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.idx == 0 or self.full

    def sample(self, batch_size, replacement=False, transform=True):
        real_size = len(self)
        if real_size == 0:
            raise ValueError("Cannot sample from an empty BehaviorCloningReplayBuffer.")
        if int(batch_size) < 1:
            raise ValueError("batch_size must be at least 1")

        batch_size = int(batch_size)
        if not replacement and batch_size > real_size:
            raise ValueError(
                f"Requested batch_size={batch_size} without replacement, but buffer only contains {real_size} samples."
            )

        if replacement:
            idxs = torch.randint(real_size, size=(batch_size,))
        else:
            idxs = torch.randperm(real_size)[:batch_size]

        obs = self.obs[idxs].to(device=DEVICE)
        target_vectors, target_mask = _pad_target_vector_sets(
            [self.target_vectors[int(idx)] for idx in idxs.tolist()],
            device=DEVICE,
        )

        if transform:
            obs, target_vectors, target_mask = _transform_bc_batch(
                obs,
                target_vectors,
                target_mask,
                include_z_flip=self.include_z_flip,
            )

        stop_labels = self.stop_labels[idxs].to(device=DEVICE)

        return obs, target_vectors, target_mask, stop_labels

    def __len__(self):
        return self.capacity if self.full else self.idx

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
    current_target_vectors : torch.Tensor
        Tensor to store target vectors.
    next_target_vectors : torch.Tensor
        Tensor to store next target vectors.
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

        self.obs = torch.empty((capacity, *obs_shape), dtype=torch.uint8, device='cpu')
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32, device='cpu')
        self.next_obs = torch.empty((capacity, *obs_shape), dtype=torch.uint8, device='cpu')
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device='cpu')
        self.current_target_vectors = [None] * capacity
        self.next_target_vectors = [None] * capacity
        self.dones = torch.empty((capacity, 1), dtype=torch.bool, device='cpu')

        self.idx = 0
        self.full = False
        self.capacity = capacity

    def push(self, obs, action, next_obs, reward, current_target_vectors, next_target_vectors, done):
        """Save a transition to replay memory"""
        self.obs[self.idx] = _require_uint8_observation(obs, name="obs")
        self.actions[self.idx] = action
        self.next_obs[self.idx] = _require_uint8_observation(next_obs, name="next_obs")
        self.rewards[self.idx] = reward
        self.current_target_vectors[self.idx] = _normalize_target_vectors(current_target_vectors)
        self.next_target_vectors[self.idx] = _normalize_target_vectors(next_target_vectors)
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.idx == 0 or self.full
    
    def sample(self, batch_size, replacement=False, transform=False):
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
        current_target_vectors, current_target_mask = _pad_target_vector_sets(
            [self.current_target_vectors[int(idx)] for idx in idxs.tolist()],
            device=DEVICE,
        )
        next_target_vectors, next_target_mask = _pad_target_vector_sets(
            [self.next_target_vectors[int(idx)] for idx in idxs.tolist()],
            device=DEVICE,
        )
        dones = self.dones[idxs].to(device=DEVICE)

        if transform:
            obs, actions, next_obs, current_target_vectors, current_target_mask, next_target_vectors, next_target_mask = _transform_batch(
                obs,
                actions,
                next_obs,
                current_target_vectors,
                current_target_mask,
                next_target_vectors,
                next_target_mask,
                include_z_flip=False,
            )

        return (
            obs,
            actions,
            next_obs,
            rewards,
            current_target_vectors,
            current_target_mask,
            next_target_vectors,
            next_target_mask,
            dones,
        )

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
    target_vectors : torch.Tensor
        A tensor to store target vectors.
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

        self.obs = torch.empty((capacity, *obs_shape), dtype=torch.uint8, device='cpu')
        self.actions = torch.empty((capacity, *action_shape), dtype=torch.float32, device='cpu')
        self.next_obs = torch.empty((capacity, *obs_shape), dtype=torch.uint8, device='cpu')
        self.rewards = torch.empty((capacity, 1), dtype=torch.float32, device='cpu')
        self.current_target_vectors = [None] * capacity
        self.next_target_vectors = [None] * capacity
        self.dones = torch.empty((capacity, 1), dtype=torch.bool, device='cpu')

        self.idx = 0
        self.full = False
        self.capacity = capacity
    
    def push(self, obs, action, next_obs, reward, current_target_vectors, next_target_vectors, done):
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
        current_target_vectors : object
            The current target vector candidates associated with the transition.
        next_target_vectors : object
            The next target vector candidates associated with the transition.
        done : bool
            Whether the episode has ended.
        """

        self.tree.add(self.max_priority, self.idx)

        self.obs[self.idx] = _require_uint8_observation(obs, name="obs")
        self.actions[self.idx] = action
        self.next_obs[self.idx] = _require_uint8_observation(next_obs, name="next_obs")
        self.rewards[self.idx] = reward
        self.current_target_vectors[self.idx] = _normalize_target_vectors(current_target_vectors)
        self.next_target_vectors[self.idx] = _normalize_target_vectors(next_target_vectors)
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.idx == 0 or self.full
        

    def sample(self, batch_size, transform: bool=False):
        """
        Samples a batch of transitions from the buffer.
        
        Parameters
        ----------
        batch_size : int
            The number of transitions to sample.
        transform : bool
            Whether to randomly flip and permute images and actions. Default is False.
            
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
        current_target_vectors : torch.Tensor
            The current target vectors of the sampled transitions.
        next_target_vectors : torch.Tensor
            The next target vectors of the sampled transitions.
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
        
        # Handle edge case where tree.total might be 0
        if self.tree.total <= 0:
            raise ValueError(f"Tree total is {self.tree.total}, which should not happen. Buffer may be corrupted.")
        
        sample_idxs, tree_idxs = [], []
        priorities = torch.empty(batch_size, 1, dtype=torch.float)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)

            cumsum = random.uniform(a, b)
            # Ensure cumsum doesn't exceed tree.total due to floating point precision
            cumsum = min(cumsum, self.tree.total)
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
        current_target_vectors, current_target_mask = _pad_target_vector_sets(
            [self.current_target_vectors[int(idx)] for idx in sample_idxs],
            device=DEVICE,
        )
        next_target_vectors, next_target_mask = _pad_target_vector_sets(
            [self.next_target_vectors[int(idx)] for idx in sample_idxs],
            device=DEVICE,
        )
        dones = self.dones[sample_idxs].to(DEVICE)

        if transform:
            obs, actions, next_obs, current_target_vectors, current_target_mask, next_target_vectors, next_target_mask = _transform_batch(
                obs,
                actions,
                next_obs,
                current_target_vectors,
                current_target_mask,
                next_target_vectors,
                next_target_mask,
                include_z_flip=True,
            )

        return (
            obs,
            actions,
            next_obs,
            rewards,
            current_target_vectors,
            current_target_mask,
            next_target_vectors,
            next_target_mask,
            dones,
            weights,
            tree_idxs,
        )
    

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
            
            # Check for NaN or invalid values
            if not isinstance(priority, (int, float)) or np.isnan(priority) or np.isinf(priority):
                print(f"Warning: Invalid priority {priority} detected, using eps instead")
                priority = self.eps
            
            # Ensure priority is non-negative before transformation
            priority = abs(priority) + self.eps
            priority = priority ** self.alpha
            
            # Check again after transformation
            if np.isnan(priority) or np.isinf(priority):
                print(f"Warning: Priority became invalid after transformation, using eps instead")
                priority = self.eps
                
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    
    def __len__(self):
        return self.capacity if self.full else self.idx
    
if __name__ == "__main__":
    pass