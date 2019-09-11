import torch
import random
import numpy as np
from collections import namedtuple, deque
from .segment_tree import SumSegmentTree, MinSegmentTree
from .sum_tree import SumTree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state",
                                     "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


# Credit: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
class PrioritizedReplayBuffer(ReplayBuffer):

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.6):
        """ Create Prioritized Replay Buffer.
            New Parameter
            ----------
            alpha: float
                how much prioritization is used
                (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(action_size, buffer_size,
                                                      batch_size, seed)
        assert alpha >= 0
        self.alpha = alpha
        self.tree = SumTree(buffer_size)
        self.priorities = [self.tree.get_val(i)**-self.alpha for i in
                           range(self.tree.filled_size())]
        # self.max_priority = 1.0
        # self.next_idx = 0
        # self.pos = 0
        # self.priorities = np.zeros((buffer_size,), dtype=np.float32)
        # tree_capacity = 1
        # while tree_capacity < buffer_size:
        #     tree_capacity *= 2
        # self.sum_tree = SumSegmentTree(tree_capacity)
        # self.min_tree = MinSegmentTree(tree_capacity)

    def add(self, state, action, reward, next_state, done, priority=1):
    # def add(self, priority=1, *args, **kwargs):
        """Add a new experience to memory."""
        # super().add(*args, **kwargs)
        e = self.experience(state, action, reward, next_state, done)
        # self.memory.append(e)
        self.tree.add(e, priority**self.alpha)
        # idx = self.next_idx
        # self.sum_tree[idx] = self.max_priority ** self.alpha
        # self.min_tree[idx] = self.max_priority ** self.alpha
        # self.next_idx = (self.next_idx + 1) % len(self.memory)
        # self.max_priority = self.priorities.max() if self.memory else self.max_priority

    # def sample_proportional(self):
    #     res = []
    #     p_total = self.sum_tree.sum(0, len(self.memory) - 1)
    #     every_range_len = p_total / self.batch_size
    #     for i in range(self.batch_size):
    #         mass = random.random() * every_range_len + i * every_range_len
    #         idx = self.sum_tree.find_prefixsum_idx(mass)
    #         res.append(idx)
    #     return res

    def sample(self, beta=0.4):
        """Sample a batch of experiences.
        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        weights: np.array
            Array of shape (batch_size,) and dtype np.float32
            denoting importance weight of each sampled transition
        idxes: np.array
            Array of shape (batch_size,) and dtype np.int32
            idexes in buffer of sampled experiences
        """
        assert beta > 0
        out = []
        indices = []
        weights = []
        self.priorities = []
        for _ in range(self.batch_size):
            r = random.random()
            data, priority, index = self.tree.find(r)
            self.priorities.append(priority)
            weights.append((1./len(self.memory)/priority)**beta if priority > 1e-16 else 0)
            indices.append(index)
            out.append(data)
            self.priority_update([index], [0]) # To avoid duplicating

        self.priority_update(indices, self.priorities) # Revert priorities
        weights = np.array(weights)/max(weights) # Normalize for stability


        # TO DO: compare self.tree with self.memory experience to see whether
        #        self.memory is even required

        # assert beta > 0
        # idxes = self.sample_proportional()
        # weights = []
        # p_min = self.min_tree.min() / self.sum_tree.sum()
        # max_weight = (p_min * len(self.memory)) ** (-beta)
        #
        # for idx in idxes:
        #     p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        #     weight = (p_sample * len(self.memory)) ** (-beta)
        #     weights.append(weight / max_weight)
        # priotity weights
        # p_weights = np.array(weights)
        # encoded_sample = self._encode_sample(idxes)
        # return tuple(list(encoded_sample) + [weights, idxes])
        # priority memory that includes include weights of each experiences
        # p_memory = map(self.memory, p_weights)

        experiences = random.sample(self.memory, k=self.batch_size)
        # experiences = random.choice(self.memory, k=self.batch_size, p=weights)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def priority_update(self, indices, priorities):
        """ The methods update samples's priority.

        Parameters
        ----------
        indices :
            list of sample indices
        """
        for i, p in zip(indices, priorities):
            self.tree.val_update(i, p**self.alpha)

    # def update_priorities(self, idxes, priorities):
    #     """Update priorities of sampled transitions.
    #     sets priority of transition at index idxes[i] in buffer
    #     to priorities[i].
    #     Parameters
    #     ----------
    #     idxes: [int]
    #         List of idxes of sampled transitions
    #     priorities: [float]
    #         List of updated priorities corresponding to
    #         transitions at the sampled idxes denoted by
    #         variable `idxes`.
    #     """
    #     assert len(idxes) == len(priorities)
    #     for idx, priority in zip(idxes, priorities):
    #         assert priority > 0
    #         assert 0 <= idx < len(self.memory)
    #         self.sum_tree[idx] = priority ** self.alpha
    #         self.min_tree[idx] = priority ** self.alpha
    #
    #         self.max_priority = max(self.max_priority, priority)
