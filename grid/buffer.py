import torch
import random
import numpy as np

from tree import SumTree



class PrioritizedReplayBuffer:
    def __init__(self, max_size, eps=1e-6, alpha=0.5, beta=0.4, beta_steps=None, tl=None):
        self.tree = SumTree(size=max_size)

        # PER params
        self.eps = eps  # minimal priority, prevents zero probabilities
        self.alpha = alpha  # determines how much prioritization is used, α = 0 corresponding to the uniform case
        #self.alpha = 0
        self.beta = beta  # determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.max_priority = 1.  # priority for new samples, init as eps
        self.tl = tl
        if beta_steps == 0:
            self.beta_add = 0
        else:
            self.beta_add = (1.0 - beta) / beta_steps
        self.alpha_add = alpha / 1000.
        # transition: state, action, reward, next_state, done
        self.states = [[]] * max_size
        self.outcomes = [[]] * max_size
        self.actions = [[]] * max_size
        self.rewards = np.zeros((max_size,), dtype=np.float32)
       
        self.episode_lens = np.zeros((max_size,), dtype=np.int)
        """back buffer"""
        self.b_states = [[]] * max_size
        self.b_outcomes = [[]] * max_size
        self.b_actions = [[]] * max_size
        self.b_rewards = np.zeros((max_size,), dtype=np.float32)
        self.b_episode_lens = np.zeros((max_size,), dtype=np.int)

        self.ptr, self.b_ptr = 0, 0
        self.real_size = 0
        self.size = max_size

    def add(self, states, outcomes, actions, reward, episode_len, if_back=False):
        # store transition index with maximum priority in sum tree

        # store transition in the buffer
        if if_back:
            for i in range(len(states)):
                self.b_states[self.b_ptr] = states[i].cpu().numpy()
                self.b_actions[self.b_ptr] = actions[i].cpu().numpy()
                self.b_episode_lens[self.b_ptr] = episode_len[i]
                self.b_outcomes[self.b_ptr] = outcomes[i].cpu().numpy()
                self.b_rewards[self.b_ptr] = reward[i]  # .cpu().numpy()
                self.b_ptr = (self.b_ptr + 1) % self.size

        else:
            for i in range(len(states)):
                self.tree.add(self.max_priority, self.ptr)
                self.states[self.ptr] = states[i].cpu().numpy()
                self.actions[self.ptr] = actions[i].cpu().numpy()
                self.episode_lens[self.ptr] = episode_len[i]
                self.outcomes[self.ptr] = outcomes[i].cpu().numpy()
                self.rewards[self.ptr] = reward[i]  # .cpu().numpy()
                self.ptr = (self.ptr + 1) % self.size
                self.real_size = min(self.size, self.real_size + 1)


    def sample(self, batch_size):
        if batch_size >= self.real_size:
            ind = np.array([i for i in range(self.real_size)])
            batch_size = self.real_size
        else:
            ind = np.random.randint(0, self.real_size, size=batch_size)
       
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

        
        weights = (self.real_size * probs) ** -self.beta

       
        weights = weights / weights.max()
       
       
        b_sample_idxs = sample_idxs.copy()
 
        batch = (
            [self.tl(self.states[ind_]) for ind_ in sample_idxs] + [self.tl(self.b_states[ind_]) for ind_ in b_sample_idxs],
            [self.tl(self.outcomes[ind_]) for ind_ in sample_idxs] + [self.tl(self.b_outcomes[ind_]) for ind_ in b_sample_idxs],
            [self.tl(self.actions[ind_]) for ind_ in sample_idxs] + [self.tl(self.b_actions[ind_]) for ind_ in b_sample_idxs],
            self.tl(np.concatenate([self.rewards[sample_idxs],self.b_rewards[b_sample_idxs]], axis=0)),
            np.concatenate([self.episode_lens[sample_idxs],self.b_episode_lens[b_sample_idxs]], axis=0),
        )
      
        return batch, weights, tree_idxs, len(sample_idxs)
    def random_sample(self, batch_size):
        if batch_size >= self.ptr:
            ind = np.array([i for i in range(self.real_size)])
            batch_size = self.real_size
        else:
            ind = np.array([i for i in range(self.ptr-batch_size,self.ptr)])#np.random.randint(0, self.real_size, size=batch_size)
       
        sample_idxs = ind
        b_sample_idxs = sample_idxs.copy()
    
        batch = (
            [self.tl(self.states[ind_]) for ind_ in sample_idxs] + [self.tl(self.b_states[ind_]) for ind_ in b_sample_idxs],
            [self.tl(self.outcomes[ind_]) for ind_ in sample_idxs] + [self.tl(self.b_outcomes[ind_]) for ind_ in b_sample_idxs],
            [self.tl(self.actions[ind_]) for ind_ in sample_idxs] + [self.tl(self.b_actions[ind_]) for ind_ in b_sample_idxs],
            self.tl(np.concatenate([self.rewards[sample_idxs],self.b_rewards[b_sample_idxs]], axis=0)),
            np.concatenate([self.episode_lens[sample_idxs],self.b_episode_lens[b_sample_idxs]], axis=0),
        )
      
        return batch, None, None, len(sample_idxs)   
    def sample_normal(self, batch_size):
        if batch_size >= self.real_size:
            ind = np.array([i for i in range(self.real_size)])
            batch_size = self.real_size
        else:
            ind = np.random.randint(0, self.real_size, size=batch_size)

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

       
        weights = (self.real_size * probs) ** -self.beta

     
        weights = weights / weights.max()
        sample_idxs = np.random.randint(0, self.real_size, size=batch_size)
        batch = (
            [self.tl(self.states[ind_]) for ind_ in sample_idxs],
            [self.tl(self.outcomes[ind_]) for ind_ in sample_idxs],
            [self.tl(self.actions[ind_]) for ind_ in sample_idxs],
            self.tl(self.rewards[sample_idxs]),
            self.episode_lens[sample_idxs],
        )

        self.beta = min(1.0, self.beta + self.beta_add)
        return batch, weights, tree_idxs, None

    def update_priorities(self, data_idxs, priorities):
        # if isinstance(priorities, torch.Tensor):
        #     priorities = priorities.detach().cpu().numpy()
        for data_idx in data_idxs:
            ### A simple way to ensure age-based sampling
            priority = 0
            

            priority = (priority + self.eps) ** self.alpha
            
            self.tree.update(data_idx, priority)
            self.max_priority = max(self.max_priority, priority)


