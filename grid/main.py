import argparse
import copy
import gzip
import heapq
import itertools
import os
import pickle
from collections import defaultdict
from itertools import count

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

import wandb
import random
import sys
import tempfile
import datetime
from itertools import chain
from buffer import  PrioritizedReplayBuffer
import time

import h5py

from multiprocessing import Pool, Manager, Process
import copy

from scipy.stats import entropy
from enum import Enum
##distribution
from torch.distributions import kl, categorical
# torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()

parser.add_argument("--save_path", default='results/flow_insp_0.pkl.gz', type=str)
parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--progress", action='store_true')
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--save_model", default=1, type=int)
parser.add_argument("--dir", default='./results', type=str)
parser.add_argument("--task", default='train', type=str)

parser.add_argument("--env", default='grid', type=str)

#
parser.add_argument("--method", default='flownet', type=str)
parser.add_argument("--learning_rate", default=1e-4, help="Learning rate", type=float)
parser.add_argument("--tb_lr", default=0.001, help="Learning rate", type=float)
parser.add_argument("--tb_z_lr", default=0.1, help="Learning rate", type=float)
parser.add_argument("--opt", default='adam', type=str)
parser.add_argument("--adam_beta1", default=0.9, type=float)
parser.add_argument("--adam_beta2", default=0.999, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--mbsize", default=16, help="Minibatch size", type=int)
parser.add_argument("--train_to_sample_ratio", default=1, type=float)
parser.add_argument("--n_hid", default=256, type=int)
parser.add_argument("--n_layers", default=2, type=int)
parser.add_argument("--n_train_steps", default=20000, type=int)
parser.add_argument("--num_empirical_loss", default=200000, type=int,
                    help="Number of samples used to compute the empirical distribution loss")

# Env
parser.add_argument('--func', default='corners')
parser.add_argument("--horizon", default=8, type=int)
parser.add_argument("--ndim", default=2, type=int)
parser.add_argument("--max_episode_len", default=-1, type=int)

parser.add_argument("--bufsize", default=16, help="MCMC buffer size", type=int)
parser.add_argument("--batch_size", default=16, help="learning batch size", type=int)
# Flownet
parser.add_argument("--bootstrap_tau", default=0., type=float)
parser.add_argument("--replay_strategy", default='none', type=str)  # top_k none
# parser.add_argument("--replay_sample_size", default=2, type=int)
# parser.add_argument("--replay_buf_size", default=100, type=float)
parser.add_argument("--target_update_frequency", default=4, type=int)
parser.add_argument("--tau", default=1., type=float)
parser.add_argument("--p_alpha", default=0.5, type=float)
parser.add_argument("--p_beta", default=0.4, type=float)
parser.add_argument("--beta_steps", default=0, type=int)
### regularize
parser.add_argument("--reg", default=1e-11, type=float)
parser.add_argument("--decay_beta", default=0.999, type=float)
parser.add_argument("--clip_grad_norm", default=0., type=float)

parser.add_argument("--wdb", action='store_true')
parser.add_argument("--exp_name", default='', type=str)

parser.add_argument("--record_traj", default=0, type=int)
parser.add_argument("--outdir", default='', type=str)

parser.add_argument("--exp_weight", default=0., type=float)
parser.add_argument("--temp", default=1., type=float)

parser.add_argument("--rand_pb", default=0, type=int)  # whether to use uniform P_B

parser.add_argument("--R0", default=1e-1, type=float)
parser.add_argument("--com_l1", default=1, type=int)
parser.add_argument("--category", default='train', type=str)
parser.add_argument("--load_steps", default='', type=str)

parser.add_argument("--her", default=0, type=int)
parser.add_argument("--backward", default=0, type=int)

parser.add_argument("--fl", default=0, type=int)
parser.add_argument("--succ_thres", default=-1, type=float)

_dev = [torch.device('cuda')]
tf = lambda x: torch.FloatTensor(x).to(_dev[0])
tl = lambda x: torch.LongTensor(x).to(_dev[0])


def set_device(dev):
    _dev[0] = dev


def func_corners(x, R0=1e-1):
    ax = abs(x)
    r = (ax > 0.5).prod(-1) * 0.5 + ((ax < 0.8) * (ax > 0.6)).prod(-1) * 2
    r += R0
    return r


class GridEnv:
    def __init__(self, horizon, ndim=2, xrange=[-1, 1], func=None, print_r=False, R0=1e-1, func_name='corners',
                 mode_list=None):
        self.horizon = horizon
        self.ndim = ndim
        assert self.ndim == 2
        self.func = func
        self.R0 = R0
        self.xspace = np.linspace(*xrange, horizon)  # split the xrange into #horizon pieces
        self.func_name = func_name

        rs = []
        smode_ls = []
        if self.ndim == 2:
            for i in range(self.horizon):
                if print_r:
                    if i == 0:
                        print('-' * 33)
                for j in range(self.horizon):
                    if print_r:
                        if j == 0:
                            print('|', end='')
                    s = np.int32([i, j])
                    r = self.func(self.s2x(s), R0=self.R0)
                    m = self.s2mode(s)
                    if print_r:
                        print(r, end='|')
                    rs.append(r)
                    smode_ls.append(m)
                if print_r:
                    print()
                    print('-' * 33)

        self.rs = rs
        rs = np.array(rs)
        self.true_density = rs / rs.sum()

    # [1, 6, 3] -> [1, -1, 0]
    # 1 or -1 means in mode, 0 means not in mode
    def s2mode(self, s):
        ret = np.int32([0] * self.ndim)
        x = self.s2x(s)
        ret += np.int32((x > 0.6) * (x < 0.8))
        ret += -1 * np.int32((x > -0.8) * (x < -0.6))
        # print (s, ret)
        return ret

    def obs(self, s=None):
        # convert s into z (index in n_dim array)
        s = np.int32(self._state if s is None else s)
        z = np.zeros((self.horizon * self.ndim), dtype=np.float32)
        z[np.arange(len(s)) * self.horizon + s] = 1
        return z

    def s2x(self, s):
        # convert index into real coordinate
        x = (self.obs(s).reshape((self.ndim, self.horizon)) * self.xspace[None, :]).sum(1)
        return x

    def reset(self):
        # return obs, reward, _state
        self._state = np.int32([0] * self.ndim)
        self._step = 0
        rew = self.func(self.s2x(self._state), R0=self.R0)
        return self.obs(), rew, self._state

    def step(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0

        # take a step in the corresponding axis [a] if not stop
        if a < self.ndim:
            s[a] += 1

        done = (a == self.ndim)

        if _s is None:
            self._state = s
            self._step += 1

        rew = self.func(self.s2x(s), R0=self.R0) if done else 0

        return self.obs(s), rew, done, s

    def reset_back(self, target_s):
        # return obs, reward, _state
        self._state = np.int32(target_s)
        self._step = 0
        rew = self.func(self.s2x(self._state), R0=self.R0)
        return self.obs(), rew, self._state

    def step_back(self, a, s=None):
        _s = s
        s = (self._state if s is None else s) + 0

        # take a step in the corresponding axis [a] if not stop
        if a < self.ndim:
            s[a] -= 1

        done = (s == 0).prod()

        if _s is None:
            self._state = s
            self._step += 1

        return self.obs(s), done, s


def make_mlp(l, act=nn.LeakyReLU(), tail=[]):
    """makes an MLP with no top layer activation"""
    return nn.Sequential(*(
                sum([[nn.Linear(i, o)] + ([act] if n < len(l) - 2 else []) for n, (i, o) in enumerate(zip(l, l[1:]))],
                    []) + tail))


class TrajBuffer(object):
    def __init__(self, state_dim, max_episode_len, max_size=int(1e5)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        #self.states = np.zeros((max_size, max_episode_len, state_dim), dtype=np.float32)
        self.states = [[]] * max_size
        self.outcomes = [[]] * max_size
        self.actions = [[]] * max_size
        self.rewards = np.zeros((max_size,), dtype=np.float32)
        #self.actions = np.zeros((max_size, max_episode_len, 1), dtype=np.float32)
        self.episode_lens = np.zeros((max_size,), dtype=np.int)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, states, outcomes, actions, reward, episode_len):
        for i in range(len(states)):
            self.states[self.ptr] = states[i].cpu().numpy()
            self.actions[self.ptr] = actions[i].cpu().numpy()
            self.episode_lens[self.ptr] = episode_len[i]
            self.outcomes[self.ptr] = outcomes[i].cpu().numpy()
            self.rewards[self.ptr] = reward[i]#.cpu().numpy()

            self.ptr = (self.ptr + 1) % self.max_size
            self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if batch_size >= self.size:
            ind = np.array([i for i in range(self.size)])
        else:
            ind = np.random.randint(0, self.size, size=batch_size)

        sampled_data = (
            [tl(self.states[ind_]) for ind_ in ind],
            [tl(self.outcomes[ind_]) for ind_ in ind],
            [tl(self.actions[ind_]) for ind_ in ind],
            tl(self.rewards[ind]),
            self.episode_lens[ind],
        )
        return sampled_data

    def save(self, fname):
        dataset = h5py.File(fname, 'w')
        dataset.create_dataset('states', data=self.states[:self.size], compression='gzip')
        dataset.create_dataset('actions', data=self.actions[:self.size], compression='gzip')
        dataset.create_dataset('episode_lens', data=self.episode_lens[:self.size], compression='gzip')

    def load(self, fname):
        data_dict = {}
        with h5py.File(fname, 'r') as dataset_file:
            for k in dataset_file.keys():
                if k == 'states':
                    self.states = dataset_file[k][:]
                elif k == 'actions':
                    self.actions = dataset_file[k][:]
                elif k == 'episode_lens':
                    self.episode_lens = dataset_file[k][:]
        self.size = self.states.shape[0]

        self.outcomes = np.zeros_like(self.states, dtype=np.float32)
        for sample_idx in range(self.size):
            curr_steps = int(self.episode_lens[sample_idx])
            y = self.states[sample_idx][curr_steps - 1]
            y = np.expand_dims(y, axis=0)
            y = np.repeat(y, curr_steps, axis=0)
            self.outcomes[sample_idx][:curr_steps] = y

        print('dataset loaded from {}: s {}, y: {}, a {}, len: {}'.format(fname, self.states.shape, self.outcomes.shape,
                                                                          self.actions.shape, self.episode_lens.shape))
def sticky_based_action_modification(states, acts, stick, dev, action_dim, horizon):
    action_dim -= 1
    rand_probs = np.random.rand(acts.shape[0])
    flags = torch.tensor(rand_probs < stick).long().to(dev)
    flags1 = (acts != action_dim).long()
    flags *= flags1
    flags2 = (states == horizon - 1).sum(-1)
    flags *= (1 - flags2)
    rand_acs = torch.tensor(np.random.randint(0, action_dim, rand_probs.shape)).long().to(dev)
    sticky_acs = acts * (1 - flags) + rand_acs * flags
    return sticky_acs
def sticky_based_action_modification_back(states, acts, stick, dev, action_dim, horizon):
    action_dim -= 1
    rand_probs = np.random.rand(acts.shape[0])
    flags = torch.tensor(rand_probs < stick).long().to(dev)
    flags1 = (acts != action_dim).long()
    flags *= flags1
    flags2 = (states == 0).sum(-1)
    flags *= (1 - flags2)
    rand_acs = torch.tensor(np.random.randint(0, action_dim, rand_probs.shape)).long().to(dev)
    sticky_acs = acts * (1 - flags) + rand_acs * flags
    return sticky_acs
class GFNModel(nn.Module):
    def __init__(self, state_dim, cond_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim + cond_dim, args.n_hid),
            nn.LeakyReLU(),
            nn.Linear(args.n_hid, args.n_hid)
        )
        self.back = nn.Sequential(
            nn.Linear(args.n_hid, args.n_hid),
            nn.LeakyReLU(),
            nn.Linear(args.n_hid, args.ndim)
        )
        self.forw= nn.Sequential(
            nn.Linear(args.n_hid, args.n_hid),
            nn.LeakyReLU(),
            nn.Linear(args.n_hid, args.ndim+2)
        )
    def forward(self, x):
        
        share = self.fc(x)
        p_f = self.forw(share)
        p_b = self.back(share)
        return p_f[:, :-1], p_b, p_f[:, -1]
class ConditionalDBFlowNetAgent:
    def __init__(self, args, envs):
        state_dim = args.horizon * args.ndim
        cond_dim = args.horizon * args.ndim

        # P_F: ndim + 1, P_B: ndim, F: 1
        #self.model = make_mlp([state_dim + cond_dim] + [args.n_hid] * args.n_layers + [2 * args.ndim + 2])
        self.model = GFNModel(state_dim, cond_dim)
        self.model.to(args.dev)
        print('model', self.model)

        self.target_model = copy.deepcopy(self.model)
       
        self.rng = np.random.default_rng(args.seed)
        self.envs = envs

        self.ndim = args.ndim
        self.horizon = args.horizon
        self.dev = args.dev

        self.exp_weight = args.exp_weight
        self.temp = args.temp
        self.uniform_pb = args.rand_pb

        self.record_traj = args.record_traj
        self.outdir = args.outdir

        self.iter_cnt = 0

        self.com_l1 = args.com_l1
        self.fl = args.fl

    def parameters(self):
        return self.model.parameters()

    def soft_update_params(self, tau):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    def print_step_stats(self, s, acts, step):
        formatted_s = torch.nonzero(s == 1)[:, 1].reshape(-1, 2)
        formatted_s[:, 1] -= 8
        s_a = torch.cat((formatted_s, acts.unsqueeze(-1)), -1).cpu().data.numpy()
        formatted_r = np.array([r for _, r, _, _ in step])
        formatted_sar = np.concatenate((s_a, formatted_r[:, np.newaxis]), -1)
        print('\033[32m{}\033[0m'.format(formatted_sar))
    def evaluate(self, to_print=False):
        all_visited = []
        self.iter_cnt += 1
        visited_goals = []

        mbsize = min(500, self.horizon*self.horizon)
        self.eval_envs = [GridEnv(self.horizon, self.ndim, func=func_corners, R0=args.R0) for i in range(mbsize)]
        conditional_inputs = torch.from_numpy((self.rng.integers(low=0, high=self.horizon, size=(mbsize, self.ndim)))).to(_dev[0]).long()
        desired_goals = conditional_inputs.cpu().numpy()
        batch_s, batch_a = [[] for i in range(mbsize)], [[] for i in range(mbsize)]
        env_idx_done_map = {i: False for i in range(mbsize)}
        not_done_envs = [i for i in range(mbsize)]
        env_idx_return_map = {}

        s = tf([i.reset()[0] for i in self.eval_envs])
        # state is onehot
        if conditional_inputs is not None:
            all_y = self.convert_states_to_onehot(conditional_inputs)
            y = self.convert_states_to_onehot(conditional_inputs)
        else:
            all_y = torch.zeros(mbsize, self.horizon * self.ndim).to(self.dev)
            y = torch.zeros(mbsize, self.horizon * self.ndim).to(self.dev)
        done = [False] * mbsize

        terminals = []
        while not all(done):
            with torch.no_grad():
                s_with_y = torch.cat((s, y), -1)
               

                p_f, p_b, f = self.model(s_with_y)
                z = torch.where(s > 0)[1].reshape(s.shape[0], -1)
                z[:, 1] -= self.horizon  # z represents the coordinate of goal
                if self.ndim > 2:
                    for dim_idx in range(2, z.shape[1]):
                        z[:, dim_idx] -= self.horizon * dim_idx  # to obtain the true coordinate

                # mask unavailable actions
                edge_mask = torch.cat(
                    [(z == self.horizon - 1).float(), torch.zeros((len(done) - sum(done), 1), device=args.dev)], 1)
             
                logits = (p_f - 1000000000 * edge_mask).log_softmax(1)  # obtain logits of P_F

                sample_ins_probs = (1 - self.exp_weight) * (logits / self.temp).softmax(1) + self.exp_weight * (
                        1 - edge_mask) / (1 - edge_mask + 0.0000001).sum(1).unsqueeze(1)
                acts = sample_ins_probs.multinomial(1)
                acts = acts.squeeze(-1)

            # observation, reward, done, state
            step = [i.step(a) for i, a in zip([e for d, e in zip(done, self.eval_envs) if not d], acts)]
            if to_print:
                self.print_step_stats(s, acts, step)

            for dat_idx, (curr_s, curr_a) in enumerate(zip(s, acts)):
                env_idx = not_done_envs[dat_idx]

                curr_formatted_s = torch.where(curr_s > 0)[0]
                curr_formatted_s[1] -= self.horizon  # obtain the coordinate
                if self.ndim > 2:
                    for dim_idx in range(2, curr_formatted_s.shape[0]):
                        curr_formatted_s[dim_idx] -= self.horizon * dim_idx  # obtain the coordinate of other dims

                batch_s[env_idx].append(curr_formatted_s)
                batch_a[env_idx].append(curr_a.unsqueeze(-1))

            for dat_idx, (ns, r, d, _) in enumerate(step):
                env_idx = not_done_envs[dat_idx]
                env_idx_done_map[env_idx] = d.item()

                if d.item():
                    env_idx_return_map[env_idx] = r.item()

                    formatted_ns = np.where(ns > 0)[0]
                    formatted_ns[1] -= self.horizon
                    if self.ndim > 2:
                        for dim_idx in range(2, formatted_ns.shape[0]):
                            formatted_ns[dim_idx] -= self.horizon * dim_idx
                    formatted_ns = formatted_ns.tolist()
                    batch_s[env_idx].append(tl(formatted_ns))

            not_done_envs = []
            for env_idx in env_idx_done_map:
                if not env_idx_done_map[env_idx]:
                    not_done_envs.append(env_idx)

            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])

            y = []
            for env_idx in not_done_envs:
                y.append(all_y[env_idx])
            if len(y) > 0:
                y = torch.stack(y)

          
            for (_, r, d, sp) in step:
                if d:
                    if self.com_l1:
                        converted_sp = sp[0] * self.horizon + sp[1]
                        if self.ndim > 2:
                            # ndim = 3
                            converted_sp = converted_sp * self.horizon + sp[2]
                            if self.ndim == 4:
                                converted_sp = converted_sp * self.horizon + sp[3]
                        all_visited.append(converted_sp)
                    else:
                        all_visited.append(tuple(sp))
                    terminals.append(list(sp))

        batch_steps = [len(batch_s[i]) for i in range(len(batch_s))]

        for i in range(len(batch_s)):
            batch_s[i] = torch.stack(batch_s[i])
            batch_a[i] = torch.stack(batch_a[i])

            assert batch_s[i].shape[0] - batch_a[i].shape[0] == 1

        batch_R = []
        for i in range(len(batch_s)):
            batch_R.append(env_idx_return_map[i])

        if self.record_traj:
            for terminal in terminals:
                curr_s_str = str(terminal)[1:-1]
                with open(os.path.join(self.outdir, 'trajectories.txt'), 'a+') as f:
                    f.writelines(curr_s_str + '\n')

        formatted_conditional_inputs = conditional_inputs.cpu().data.numpy().tolist()
        for i in range(len(batch_s)):
            curr_data = []
            curr_data.extend(formatted_conditional_inputs[i])
            curr_traj = batch_s[i].cpu().data.numpy().tolist()
            for j in range(len(curr_traj)):
                curr_data.extend(curr_traj[j])
            curr_data_str = str(curr_data)[1:-1]
            with open(os.path.join(self.outdir, 'trajectories.txt'), 'a+') as f:
                f.writelines(curr_data_str + '\n')
            # print (i, curr_data_str)
        for i, x in enumerate(batch_R):
            if x == 1:
                visited_goals.append(batch_s[i][-1])
        if not os.path.exists("./analysis_goals"):
            # if the demo_folder directory is not present
            # then create it.
            os.makedirs("./analysis_goals")
      
        return [batch_s, conditional_inputs, batch_a, batch_R, batch_steps, mbsize]
    def sample_many(self, mbsize, conditional_inputs, all_visited, to_print=False):
        #self.iter_cnt += 1

        batch_s, batch_a = [[] for i in range(mbsize)], [[] for i in range(mbsize)]
        env_idx_done_map = {i: False for i in range(mbsize)}
        not_done_envs = [i for i in range(mbsize)]
        env_idx_return_map = {}

        s = tf([i.reset()[0] for i in self.envs])
        # state is onehot
        if conditional_inputs is not None:
            all_y = self.convert_states_to_onehot(conditional_inputs)
            y = self.convert_states_to_onehot(conditional_inputs)
        else:
            all_y = torch.zeros(mbsize, self.horizon * self.ndim).to(self.dev)
            y = torch.zeros(mbsize, self.horizon * self.ndim).to(self.dev)
        done = [False] * mbsize

        terminals = []
        while not all(done):
            with torch.no_grad():
                s_with_y = torch.cat((s, y), -1)
              
                p_f, p_b, f = self.model(s_with_y)
                z = torch.where(s > 0)[1].reshape(s.shape[0], -1)
                z[:, 1] -= self.horizon #z represents the coordinate of goal
                if self.ndim > 2:
                    for dim_idx in range(2, z.shape[1]):
                        z[:, dim_idx] -= self.horizon * dim_idx #to obtain the true coordinate

                # mask unavailable actions
                edge_mask = torch.cat(
                    [(z == self.horizon - 1).float(), torch.zeros((len(done) - sum(done), 1), device=args.dev)], 1)
                #REVIEW why edge_mask*1e9
                logits = (p_f - 1000000000 * edge_mask).log_softmax(1) #obtain logits of P_F

                sample_ins_probs = (1 - self.exp_weight) * (logits / self.temp).softmax(1) + self.exp_weight * (
                            1 - edge_mask) / (1 - edge_mask + 0.0000001).sum(1).unsqueeze(1)
                acts = sample_ins_probs.multinomial(1)
                acts = acts.squeeze(-1)
            
            noisy_acts = sticky_based_action_modification(z, acts, 0.01, self.dev, self.ndim + 1, args.horizon)
            # observation, reward, done, state
            step = [i.step(a) for i, a in zip([e for d, e in zip(done, self.envs) if not d], noisy_acts)]
            if to_print:
                self.print_step_stats(s, acts, step)

            for dat_idx, (curr_s, curr_a) in enumerate(zip(s, acts)):
                env_idx = not_done_envs[dat_idx]

                curr_formatted_s = torch.where(curr_s > 0)[0]
                curr_formatted_s[1] -= self.horizon #obtain the coordinate
                if self.ndim > 2:
                    for dim_idx in range(2, curr_formatted_s.shape[0]):
                        curr_formatted_s[dim_idx] -= self.horizon * dim_idx  #obtain the coordinate of other dims

                batch_s[env_idx].append(curr_formatted_s)
                batch_a[env_idx].append(curr_a.unsqueeze(-1))

            for dat_idx, (ns, r, d, _) in enumerate(step):
                env_idx = not_done_envs[dat_idx]
                env_idx_done_map[env_idx] = d.item()

                if d.item():
                    env_idx_return_map[env_idx] = r.item()

                    formatted_ns = np.where(ns > 0)[0]
                    formatted_ns[1] -= self.horizon
                    if self.ndim > 2:
                        for dim_idx in range(2, formatted_ns.shape[0]):
                            formatted_ns[dim_idx] -= self.horizon * dim_idx
                    formatted_ns = formatted_ns.tolist()
                    batch_s[env_idx].append(tl(formatted_ns))

            not_done_envs = []
            for env_idx in env_idx_done_map:
                if not env_idx_done_map[env_idx]:
                    not_done_envs.append(env_idx)

            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][2]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[2]])

            y = []
            for env_idx in not_done_envs:
                y.append(all_y[env_idx])
            if len(y) > 0:
                y = torch.stack(y)

            for (_, r, d, sp) in step:
                if d:
                    if self.com_l1:
                        converted_sp = sp[0] * self.horizon + sp[1]
                        if self.ndim > 2:
                            # ndim = 3
                            converted_sp = converted_sp * self.horizon + sp[2]
                            if self.ndim == 4:
                                converted_sp = converted_sp * self.horizon + sp[3]
                        all_visited.append(converted_sp)
                    else:
                        all_visited.append(tuple(sp))
                    terminals.append(list(sp))

        batch_steps = [len(batch_s[i]) for i in range(len(batch_s))]

        for i in range(len(batch_s)):
            batch_s[i] = torch.stack(batch_s[i])
            batch_a[i] = torch.stack(batch_a[i])

            assert batch_s[i].shape[0] - batch_a[i].shape[0] == 1

        batch_R = []
        for i in range(len(batch_s)):
            batch_R.append(env_idx_return_map[i])

        if self.record_traj:
            for terminal in terminals:
                curr_s_str = str(terminal)[1:-1]
                with open(os.path.join(self.outdir, 'trajectories.txt'), 'a+') as f:
                    f.writelines(curr_s_str + '\n')

        formatted_conditional_inputs = conditional_inputs.cpu().data.numpy().tolist()
        for i in range(len(batch_s)):
            curr_data = []
            curr_data.extend(formatted_conditional_inputs[i])
            curr_traj = batch_s[i].cpu().data.numpy().tolist()
            for j in range(len(curr_traj)):
                curr_data.extend(curr_traj[j])
            curr_data_str = str(curr_data)[1:-1]
            with open(os.path.join(self.outdir, 'trajectories.txt'), 'a+') as f:
                f.writelines(curr_data_str + '\n')
           

        return [batch_s, batch_a, batch_R, batch_steps]

    def sample_many_backward(self, mbsize, conditional_inputs, all_visited, to_print=False):
        inf = 1000000000

        # self.iter_cnt += 1

        batch_s, batch_a = [[conditional_inputs[i]] for i in range(mbsize)], [[tl([self.ndim])] for i in range(mbsize)]
        env_idx_done_map = {i: False for i in range(mbsize)}
        not_done_envs = [i for i in range(mbsize)]
        env_idx_return_map = {}

        s_all, formatted_s = [], []
        formatted_conditional_inputs = conditional_inputs.cpu().data.numpy().tolist()
        for env_idx, env in enumerate(self.envs):
            curr_s, curr_r, curr_s_ohe = env.reset_back(formatted_conditional_inputs[env_idx])
            s_all.append(curr_s)
            formatted_s.append(curr_s_ohe)
            env_idx_return_map[env_idx] = curr_r.item()
        done = [bool((formatted_s[i] == 0).prod()) for i in range(mbsize)]
        s = []
        for i in range(mbsize):
            if done[i]:
                batch_s[i].append(tl([0, 0]))
                env_idx_done_map[i] = True
            else:
                s.append(s_all[i])
        s = tf(s)

        assert conditional_inputs is not None
        all_y = self.convert_states_to_onehot(conditional_inputs)
        y = self.convert_states_to_onehot(conditional_inputs)

        not_done_envs = []
        for env_idx in env_idx_done_map:
            if not env_idx_done_map[env_idx]:
                not_done_envs.append(env_idx)

        y = []
        for env_idx in not_done_envs:
            y.append(all_y[env_idx])
        if len(y) > 0:
            y = torch.stack(y)

        terminals = []
        while not all(done):
            with torch.no_grad():
                s_with_y = torch.cat((s, y), -1)

               
                p_f, p_b, f = self.model(s_with_y)
                # convert s to one-hot formats
                z = torch.where(s > 0)[1].reshape(s.shape[0], -1)
                z[:, 1] -= self.horizon
                if self.ndim > 2:
                    for dim_idx in range(2, z.shape[1]):
                        z[:, dim_idx] -= self.horizon * dim_idx

                init_edge_mask = (z == 0).float()  # whether it is at an initial position
                back_logits = ((0 if self.uniform_pb else 1) * p_b - inf * init_edge_mask).log_softmax(
                    1)  # steps, n_dim

                sample_ins_probs = (1 - self.exp_weight) * (back_logits / self.temp).softmax(1) + self.exp_weight * (
                            1 - init_edge_mask) / (1 - init_edge_mask + 0.0000001).sum(1).unsqueeze(1)
                back_acts = sample_ins_probs.multinomial(1)
         
            # observation, reward, done, state
            step = [i.step_back(a) for i, a in zip([e for d, e in zip(done, self.envs) if not d], back_acts)]
            if to_print:
                self.print_step_stats(s, back_acts, step)

            for dat_idx, (curr_s, curr_a) in enumerate(zip(s, back_acts)):
                env_idx = not_done_envs[dat_idx]

                curr_formatted_s = torch.where(curr_s > 0)[0]
                curr_formatted_s[1] -= self.horizon
                if self.ndim > 2:
                    for dim_idx in range(2, curr_formatted_s.shape[0]):
                        curr_formatted_s[dim_idx] -= self.horizon * dim_idx

                batch_s[env_idx].append(curr_formatted_s)
                batch_a[env_idx].append(curr_a.unsqueeze(-1))

            for dat_idx, (_, d, _) in enumerate(step):
                env_idx = not_done_envs[dat_idx]
                env_idx_done_map[env_idx] = d.item()

                if d.item():
                    batch_s[env_idx].append(tl([0, 0]))

            not_done_envs = []
            for env_idx in env_idx_done_map:
                if not env_idx_done_map[env_idx]:
                    not_done_envs.append(env_idx)

            c = count(0)
            m = {j: next(c) for j in range(mbsize) if not done[j]}
            done = [bool(d or step[m[i]][1]) for i, d in enumerate(done)]
            s = tf([i[0] for i in step if not i[1]])

            y = []
            for env_idx in not_done_envs:
                y.append(all_y[env_idx])
            if len(y) > 0:
                y = torch.stack(y)

        batch_steps = [len(batch_s[i]) for i in range(len(batch_s))]

        for i in range(len(batch_s)):
            batch_s[i] = torch.stack(batch_s[i])
            batch_a[i] = torch.stack(batch_a[i])

            batch_s[i] = torch.flip(batch_s[i], [0])
            batch_a[i] = torch.flip(batch_a[i], [0])

            assert batch_s[i].shape[0] - batch_a[i].shape[0] == 1

        batch_R = []
        for i in range(len(batch_s)):
            batch_R.append(env_idx_return_map[i])

        if self.record_traj:
            for i in range(len(batch_s)):
                curr_data = []
                curr_data.extend(formatted_conditional_inputs[i])
                curr_traj = batch_s[i].cpu().data.numpy().tolist()
                for j in range(len(curr_traj)):
                    curr_data.extend(curr_traj[j])
                curr_data_str = str(curr_data)[1:-1]
                with open(os.path.join(self.outdir, 'trajectories.txt'), 'a+') as f:
                    f.writelines(curr_data_str + '\n')

        return [batch_s, batch_a, batch_R, batch_steps]

    def convert_states_to_onehot(self, states):
        # convert to onehot format
        return torch.nn.functional.one_hot(states, self.horizon).view(states.shape[0], -1).float()

    def learn_from(self, it, batch, length=16, to_print=False, success_cnt=None, success_thres=-1, weights=None):
        inf = 1000000000
        kl_loss = nn.KLDivLoss(reduction="batchmean") #kl
        states, outcomes, actions, returns, episode_lens = batch
     
        ll_diff = []
        t_ll_diff = []
        prob_f = []
        for data_idx in range(len(states)):
            curr_episode_len = episode_lens[data_idx]
            curr_states = states[data_idx][:curr_episode_len, :]  # episode_len + 1, state_dim
            if outcomes is not None:
                curr_outcomes = outcomes[data_idx][:curr_episode_len, :]  # episode_len + 1, state_dim
            curr_actions = actions[data_idx][:curr_episode_len - 1, :]  # episode_len, action_dim
            curr_return = returns[data_idx]

            # convert state into one-hot format: steps, ndim x horizon
            curr_states_onehot = self.convert_states_to_onehot(curr_states)
            if outcomes is not None:
                curr_outcomes_onehot = self.convert_states_to_onehot(curr_outcomes)
            else:
                curr_outcomes_onehot = torch.zeros(curr_episode_len, self.horizon * self.ndim).to(self.dev)
            curr_states_and_outcomes_onehot = torch.cat((curr_states_onehot, curr_outcomes_onehot), -1)

            # get predicted forward (from 0 to n_dim) and backward logits (from n_dim to last): steps, 2 x ndim + 1
            #pred = self.model(curr_states_and_outcomes_onehot)
            p_f, p_b, f = self.model(curr_states_and_outcomes_onehot)
            #print(p_f.shape)

     
            edge_mask = torch.cat(
                [(curr_states == self.horizon - 1).float(), torch.zeros((curr_states.shape[0], 1), device=self.dev)], 1)
            logits = (p_f - inf * edge_mask).log_softmax(1)  # steps, n_dim + 1
            prob_f.append(torch.exp(logits))
            init_edge_mask = (curr_states == 0).float()  # whether it is at an initial position
            back_logits = (
                        (0 if self.uniform_pb else 1) * p_b - inf * init_edge_mask).log_softmax(
                1)  # steps, n_dim
            #regularize
            prior_dis = F.softmax(torch.ones(p_b.shape).to(p_b.device), dim=1)
            reg_kl = kl_loss(back_logits, prior_dis)
            logits = logits[:-1, :].gather(1, curr_actions).squeeze(1)
            back_logits = back_logits[1:-1, :].gather(1, curr_actions[:-1, :]).squeeze(1)

            log_flow = f  # F(s) (the last dimension)
            log_flow = log_flow[:-1]  # ignore the last state

            if self.fl:
                fl_r = curr_return.repeat(log_flow.shape[0])
                fl_r = fl_r.float() + 1e-9

            curr_return = curr_return.float() + 1e-9
            curr_ll_diff = torch.zeros(curr_states.shape[0] - 1).to(self.dev)
            curr_ll_diff += log_flow
            curr_ll_diff += logits
            curr_ll_diff[:-1] -= log_flow[1:]
            curr_ll_diff[:-1] -= back_logits
            curr_ll_diff[:-1] += args.reg * reg_kl #regularize

            curr_ll_diff[-1] -= curr_return.log()


            if self.fl and curr_return < 1:
                if success_cnt is not None and success_thres >= 0.:
                    if success_cnt >= success_thres:
                        reachability_flag = torch.prod(curr_states[:-1] <= curr_outcomes[:-1], -1)
                        reachability_flag = 1 - reachability_flag
                        curr_ll_diff[:-1] -= (fl_r[:-1].log() * reachability_flag[:-1])
                else:
                    reachability_flag = torch.prod(curr_states[:-1] <= curr_outcomes[:-1], -1)
                    reachability_flag = 1 - reachability_flag
                    curr_ll_diff[:-1] -= (fl_r[:-1].log() * reachability_flag[:-1])

            
            if curr_ll_diff.mean()<10:
                ll_diff.append(curr_ll_diff ** 2)
        
        prob_f = torch.clip(torch.cat(prob_f), min=1e-10)

        entropy_f = entropy(prob_f.cpu().detach().numpy(), axis=1)
  
        ll_diff = torch.cat(ll_diff)
        loss = ll_diff.mean()
        args.reg = args.reg*args.decay_beta


        return [loss], None, entropy_f

    def save(self, filename):
        # save model
        torch.save(self.model.state_dict(), filename + "_model")

    def load(self, filename):
        print('\033[32mload conditional_db from {}\033[0m'.format(filename))
        self.model.load_state_dict(torch.load(filename + "_model"))


def prepare_output_dir(user_specified_dir=None, argv=None, time_format='%Y%m%dT%H%M%S.%f'):
    time_str = datetime.datetime.now().strftime(time_format)
    if user_specified_dir is not None:
        if os.path.exists(user_specified_dir):
            if not os.path.isdir(user_specified_dir):
                raise RuntimeError('{} is not a directory'.format(user_specified_dir))
        outdir = os.path.join(user_specified_dir, time_str)
        if os.path.exists(outdir):
            raise RuntimeError('{} exists'.format(outdir))
        else:
            os.makedirs(outdir)
    else:
        outdir = tempfile.mkdtemp(prefix=time_str)

    with open(os.path.join(outdir, 'command.txt'), 'w') as f:
        if argv is None:
            argv = sys.argv
        f.write(' '.join(argv))

    os.makedirs(outdir + '/models')

    return outdir


def record_stats(outdir, values, category='results'):
    with open(os.path.join(outdir, '{}.txt'.format(category)), 'a+') as f:
        print('\t'.join(str(x) for x in values), file=f)


def calc_succ_cnt(formatted_conditional_converted_Rs):
    success_cnt = 0
    for sample_idx in range(len(formatted_conditional_converted_Rs)):
        curr_conditional_r = formatted_conditional_converted_Rs[sample_idx]
        if curr_conditional_r == 1:
            success_cnt += 1

    success_cnt = success_cnt * 100.0 / len(formatted_conditional_converted_Rs)
 
    return success_cnt
def curriculum_outcomes(cur_step, steps):
    candidates = [(i, j) for i in range(args.horizon) for j in range(args.horizon)]
    dist = [1 - (i + j) / (2 * args.horizon - 1) for i in range(args.horizon) for j in range(args.horizon)]
    if cur_step >= steps:
        index = np.random.choice(len(candidates), args.mbsize, replace=False)
    else:
        index = np.random.choice(len(candidates), args.mbsize, p=F.softmax(torch.tensor([i**(1 - cur_step/steps) for i in dist], dtype=torch.float64)), replace=False)
    outcomes = np.array(candidates)[index]
    return torch.from_numpy(outcomes).to(args.device).long()
def curriculum_outcomes_v1(cur_step, steps):
    cur_step = min(cur_step, steps)
    horizon = 1 + int((cur_step / steps) * (args.horizon-1))
    outcomes_for_sample = torch.randint(low=0, high=horizon, size=(args.mbsize, args.ndim))
    outcomes_for_sample = torch.LongTensor(outcomes_for_sample).to(args.device)

    return outcomes_for_sample
def train(args):
    to_print = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    if args.wdb:
        if args.exp_name != '':
            wandb.init(project='gfn_grid', name='{}_{}'.format(args.exp_name, args.seed))
        else:
            wandb.init(project='gfn_grid')
    args.dev = torch.device(args.device)
    set_device(args.dev)

    envs = [GridEnv(args.horizon, args.ndim, func=func_corners, R0=args.R0) for i in range(args.mbsize)]

    outdir = prepare_output_dir(args.dir + '/' + args.method, argv=sys.argv)
    print('\033[32moutdir: {}\033[0m'.format(outdir))
    args.outdir = outdir

    if args.method == 'db_gfn':
        agent = ConditionalDBFlowNetAgent(args, envs)
        opt = torch.optim.Adam([{'params': agent.parameters(), 'lr': args.tb_lr}])

    all_visited_conditional = []
   
    buffer = PrioritizedReplayBuffer(max_size=int(1e6), alpha=args.p_alpha, beta=args.p_beta, beta_steps=args.beta_steps, tl=tl)

    unseen_outcomes = torch.tensor(([5,5])).long()
    saved_states = []
    original_states = []
    start =time.time()
    for i in tqdm(range(args.n_train_steps + 1), disable=not args.progress):
  
       
        outcomes_for_sample = torch.randint(low=0, high=args.horizon, size=(args.mbsize, args.ndim))


        outcomes_for_sample = torch.LongTensor(outcomes_for_sample).to(args.device)
 

        conditional_experiences = agent.sample_many(args.mbsize, outcomes_for_sample, all_visited_conditional)

        # ========== update the conditional gfn by the positive/negative samples ==========
        conditional_sampled_states, conditional_sampled_steps = conditional_experiences[0], conditional_experiences[3]
       
        y_minus = torch.stack([conditional_sampled_states[sample_idx][-1] for sample_idx in range(args.mbsize)])
        conditional_converted_Rs = torch.prod(y_minus == outcomes_for_sample, -1).float()
        achieved_goals = [y_minus[ind] for ind,d in enumerate(conditional_converted_Rs) if d==1]
      
        formatted_conditional_converted_Rs = conditional_converted_Rs.int().cpu().data.numpy().tolist()
        success_cnt = calc_succ_cnt(formatted_conditional_converted_Rs)

        record_stats(outdir, [i, success_cnt], 'conditional_success_rate')

    

        conditional_outcomes = []
        for sample_idx in range(args.mbsize):
            curr_steps = conditional_sampled_steps[sample_idx]
            y_plus = outcomes_for_sample[sample_idx]
            y_plus = y_plus.unsqueeze(0).repeat(curr_steps, 1)
            conditional_outcomes.append(y_plus)

        # s, y, a, R, steps
        converted_conditional_experiences = [conditional_sampled_states, conditional_outcomes,
                                             conditional_experiences[1], conditional_converted_Rs,
                                             conditional_sampled_steps]
     
        for traj in conditional_sampled_states:
                curr_traj = np.array(traj.tolist())[1:-1]
                original_states.append(curr_traj)
        buffer.add(*converted_conditional_experiences)
  
        if args.her or args.backward:
            # obtain conditional rewards (positive)
            converted_Rs = [1. for sample_idx in range(args.mbsize)]

            if args.her:
                conditional_sampled_states, conditional_sampled_steps = conditional_experiences[0], \
                                                                        conditional_experiences[3]
                # obtain outcomes
                outcomes = []
                for sample_idx in range(len(conditional_sampled_states)):
                    curr_steps = conditional_sampled_steps[sample_idx]
                    y_plus = conditional_sampled_states[sample_idx][-1]  # the last state
                    y_plus = y_plus.unsqueeze(0).repeat(curr_steps, 1)
                    outcomes.append(y_plus)

                # prepare inputs: states, outcomes, actions, Rs, steps
                converted_experiences = [conditional_sampled_states, outcomes, conditional_experiences[1], converted_Rs,
                                         conditional_sampled_steps]
                buffer.add(*converted_experiences, if_back=True)
            elif args.backward:
                
                conditional_experiences_backward = agent.sample_many_backward(args.mbsize, outcomes_for_sample,
                                                                              all_visited_conditional)
                conditional_sampled_states, conditional_sampled_steps = conditional_experiences_backward[0], \
                                                                        conditional_experiences_backward[3]
                
                outcomes = []
                for sample_idx in range(args.mbsize):
                    curr_steps = conditional_sampled_steps[sample_idx]
                    y_plus = outcomes_for_sample[sample_idx]
                    y_plus = y_plus.unsqueeze(0).repeat(curr_steps, 1)
                    outcomes.append(y_plus)

                converted_experiences = [conditional_sampled_states, outcomes, conditional_experiences_backward[1],
                                         converted_Rs, conditional_sampled_steps]
             
                for traj in conditional_sampled_states:
                    curr_traj = np.array(traj.tolist())[1:-1]
                    saved_states.append(curr_traj)
                    
                buffer.add(*converted_experiences, if_back=True)
            if to_print:
                for j in range(args.mbsize):
                    print('[{}]'.format(j))
                    print('s', converted_experiences[0][j])
                    print('y', converted_experiences[1][j])
                    print('a', converted_experiences[2][j])
                    print('R', converted_experiences[3][j])
                    print('steps', converted_experiences[4][j])
            
            samples, weights, tree_idxs, length = buffer.sample(args.batch_size)
            conditional_losses, error, entropy_f = agent.learn_from(i, samples, length=length)
           
            conditional_losses[0].backward()
            if args.method not in ['db_gfn']:
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), args.clip_grad_norm)
            opt.step()
            opt.zero_grad()
           
            curr_conditional_losses_pos = [i.item() for i in conditional_losses]
            if args.wdb:
                wandb_dict = {
                    'success_rate': success_cnt,
                    'entropy_f': entropy_f.mean(),
                    'conditional_loss': curr_conditional_losses_pos[0],
                    'wall-clock time': time.time()-start,
                    # 'reducible_loss': error.mean(),
                }
                # if i % 1000 == 0 :
                #     wandb_dict['eval_success_rate'] = eval_success_cnt
                wandb.log(wandb_dict)

            else:
                record_stats(outdir, [i] + curr_conditional_losses_pos, 'conditional_loss')
        

        if args.save_model and (i + 1) % 1000 == 0:
            agent.save('{}/models/step_{}'.format(outdir, i + 1))
    np.savez(os.path.join(args.outdir, 'saved_traj.npz'),traj=np.array(saved_states))
    np.savez(os.path.join(args.outdir, 'original_traj.npz'),traj=np.array(original_states))
if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_num_threads(1)
    train(args)


