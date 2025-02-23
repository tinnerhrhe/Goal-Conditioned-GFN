import argparse
import gzip
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm
from scipy.stats import linregress
from scipy.stats import spearmanr

from lib.model.mlp import GFNMLP, GFNConditionalMLP,GFNConditionalMLPV1,GFNConditionalMLPV2,GFNConditionalMLPV3,GFNConditionalTCN
from polyleven import levenshtein
import random
import time

import os, sys
import tempfile
import datetime
import shutil
import wandb

import torch.nn.functional as F
from itertools import chain
import h5py
import itertools
from polyleven import levenshtein
import re
import copy
from buffer_bit import PrioritizedReplayBuffer
# torch.autograd.set_detect_anomaly(True)
import math
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", default='', type=str)
parser.add_argument("--mlp", default='', type=str)
parser.add_argument("--tcn", default=0, type=int)
parser.add_argument("--woscale", default=0, type=int)
parser.add_argument("--save_path", default='test.pkl.gz')
parser.add_argument("--oracle_difficulty", default='medium')
parser.add_argument("--batch_size", default=16, help="learning batch size", type=int)
parser.add_argument("--target_update_frequency", default=1, type=int)
parser.add_argument("--tau", default=0.005, type=float)
parser.add_argument("--p_alpha", default=0.5, type=float)
parser.add_argument("--p_beta", default=0.4, type=float)
parser.add_argument("--beta", default=1.0, type=float)
parser.add_argument("--min_beta", default=0.0001, type=float)
parser.add_argument("--beta_steps", default=0, type=int)
parser.add_argument("--reg", default=1e-11, type=float)

# Test params
parser.add_argument("--num_bits", default=1, type=int)
parser.add_argument("--seq_max_len", default=64, type=int)
parser.add_argument("--mode_max_len", default=64, type=int)
parser.add_argument("--seq_min_len", default=32, type=int)
parser.add_argument("--num_test_modes", default=60, type=int)
parser.add_argument("--seed", default=0, type=int)
parser.add_argument("--run", default=-1, type=int)
parser.add_argument("--tsf_ver", default='amortize', type=str) 
parser.add_argument("--amortize_random_action_prob", default=0.001, type=float)
parser.add_argument("--amortize_sampling_temperature", default=1., type=float) 
parser.add_argument("--use_offline_data", default=0, type=int)
parser.add_argument("--rec_train_modes", default=1, type=int)
parser.add_argument("--rec_eval_modes", default=1, type=int)
parser.add_argument("--dist_thres", default=20, type=int)
# Generator
parser.add_argument("--gen_partition_init", default=32, type=float)
parser.add_argument("--gen_learning_rate", default=0.005, type=float)
parser.add_argument("--gen_Z_learning_rate", default=1e-3, type=float)
parser.add_argument("--gen_num_iterations", default=500000, type=int)
parser.add_argument("--gen_episodes_per_step", default=16, type=int)
parser.add_argument("--gen_num_hid", default=2048, type=int)
parser.add_argument("--gen_num_layers", default=2, type=int)
parser.add_argument("--gen_reward_norm", default=1, type=float)
parser.add_argument("--gen_reward_exp", default=3, type=float)
parser.add_argument("--gen_reward_min", default=1e-8, type=float)
parser.add_argument("--gen_clip", default=10, type=float)
parser.add_argument("--gen_L2", default=1e-4, type=float)
parser.add_argument("--gen_reward_exp_ramping", default=1, type=float)
parser.add_argument("--gen_balanced_loss", default=1, type=float)
parser.add_argument("--gen_output_coef", default=1, type=float)
parser.add_argument("--gen_loss_eps", default=1e-10, type=float)
parser.add_argument("--gen_random_action_prob", default=0.001, type=float)
parser.add_argument("--gen_sampling_temperature", default=1., type=float)
parser.add_argument("--gen_leaf_coef", default=5, type=float)
parser.add_argument("--gen_data_sample_per_step", default=0, type=int)
parser.add_argument("--gen_do_pg", default=0, type=int)
parser.add_argument("--gen_pg_entropy_coef", default=1e-4, type=float)
parser.add_argument("--gen_do_explicit_Z", default=0, type=int)  # learning partition Z explicitly

parser.add_argument("--gen_model_type", default='mlp', type=str)
parser.add_argument("--cond_model_type", default='mlp', type=str)

parser.add_argument("--dir", default='./results_bit', type=str)
parser.add_argument("--algo", default='db', type=str)
parser.add_argument("--wdb", default=0, type=int)

parser.add_argument("--rnd", default=0, type=int)
parser.add_argument("--ri_out_dim", default=128, type=int)
parser.add_argument("--ri_num_hidden", default=2048, type=int)
parser.add_argument("--ri_num_layers", default=2, type=int)
parser.add_argument("--ri_coe", default=1.0, type=float)
parser.add_argument("--ri_loss_coe", default=1., type=float)

parser.add_argument("--cond_lr", default=5e-4, type=float)
parser.add_argument("--cond_clip", default=10, type=float)
parser.add_argument("--cond_L2", default=0, type=float)
parser.add_argument("--cond_random_action_prob", default=0.0005, type=float)
parser.add_argument("--cond_sampling_temperature", default=1., type=float)  # 2
parser.add_argument("--cond_arch_ver", default='v0', type=str)
parser.add_argument("--cond_num_hidden", default=2048, type=int)
parser.add_argument("--cond_num_layers", default=2, type=int)
parser.add_argument("--cond_output_coef", default=1, type=float)
parser.add_argument("--fl", default=0, type=int)
parser.add_argument("--tag", default='', type=str)
parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--r_type", default='free', type=str)
parser.add_argument("--r_free_offset", default=1e-1, type=float)

parser.add_argument("--save_data", default=0, type=int)
parser.add_argument("--save_model", default=0, type=int)
parser.add_argument("--save_interval", default=10000, type=int)
parser.add_argument("--load_dir", default='20240806T203949.984752', type=str)
parser.add_argument("--load_step", default=14000, type=int)
parser.add_argument("--sp_r", default=0, type=int)
parser.add_argument("--sp_r_thres", default=3, type=int)


class GeneratorBase(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_step(self):
        raise NotImplementedError()

    def forward(self):
        raise NotImplementedError()

    def save(self, path):
        torch.save(self.state_dict(), path)


class RND(nn.Module):
    def __init__(self, num_tokens, num_outputs, rnd_num_hidden, rnd_num_layers, max_len):
        super(RND, self).__init__()

        self.random_target_network = GFNMLP(
            num_tokens=num_tokens,
            num_outputs=num_outputs,
            num_hid=rnd_num_hidden,
            num_layers=rnd_num_layers,
            max_len=max_len,
            dropout=0,
        )
        print('random_target_network', self.random_target_network)

        self.predictor_network = GFNMLP(
            num_tokens=num_tokens,
            num_outputs=num_outputs,
            num_hid=rnd_num_hidden,
            num_layers=rnd_num_layers,
            max_len=max_len,
            dropout=0,
        )
        print('predictor_network', self.predictor_network)

    def forward(self, next_state, lens):
        random_phi_s_next = self.random_target_network.forward_rnd(next_state, None, return_all=True, lens=lens)
        predicted_phi_s_next = self.predictor_network.forward_rnd(next_state, None, return_all=True, lens=lens)
        return random_phi_s_next, predicted_phi_s_next

    def compute_intrinsic_reward(self, next_states, lens):
        random_phi_s_next, predicted_phi_s_next = self.forward(next_states, lens)

        intrinsic_reward = torch.norm(predicted_phi_s_next.detach() - random_phi_s_next.detach(), dim=-1, p=2)

        intrinsic_reward = intrinsic_reward.cpu().detach().numpy()

        return intrinsic_reward

    def compute_loss(self, next_states, lens):
        # max_len, N, s_latent_dim
        random_phi_s_next, predicted_phi_s_next = self.forward(next_states, lens)

        rnd_loss = torch.norm(predicted_phi_s_next - random_phi_s_next.detach(), dim=-1, p=2)

        return rnd_loss


class StrBuffer(object):
    def __init__(self, str_len, max_size=int(2e7)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, str_len), dtype=np.int64)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, states):
        self.states[self.ptr] = states

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        if batch_size >= self.size:
            ind = np.array([i for i in range(self.size)])
        else:
            ind = np.random.randint(0, self.size, size=batch_size)

        sampled_data = torch.LongTensor(self.states[ind]).to(self.device)
        return sampled_data

    def save(self, fname):
        dataset = h5py.File(fname, 'w')
        dataset.create_dataset('states', data=self.states[:self.size], compression='gzip')

    def load(self, fname):
        data_dict = {}
        with h5py.File(fname, 'r') as dataset_file:
            for k in dataset_file.keys():
                if k == 'states':
                    self.states = dataset_file[k][:]
        self.size = self.states.shape[0]

        print('dataset loaded from {}: {}'.format(fname, self.states.shape))


class ConditionalDBGFlowNetGenerator(GeneratorBase):
    def __init__(self, args):
        super().__init__(args)
        self.num_tokens = 2 ** args.num_bits * 2
        print(self.num_tokens, 'tokens')
        self.max_len = int(args.seq_max_len / args.num_bits)
        self.seq_max_len = args.seq_max_len
        if self.num_tokens == 8:
            self.reward_scale = 1e7
        elif self.num_tokens == 16:
            self.reward_scale = 1e25
        elif self.num_tokens == 64:
            self.reward_scale = 1e40
        if args.woscale:
            self.reward_scale = 1
        print("self.reward_scale:", self.reward_scale)
        self.cond_model_type = args.cond_model_type
        assert self.cond_model_type == 'mlp'
        if args.tcn:
            self.model = GFNConditionalTCN(
            num_tokens=self.num_tokens,
            num_outputs=self.num_tokens + 1,  # +1 for F(s),
            num_hid=args.cond_num_hidden,
            num_layers=args.cond_num_layers,
            max_len=self.max_len,
            dropout=0,
            arch=args.cond_arch_ver
        )
        else:
            if args.mlp == 'v1':
                self.model = GFNConditionalMLPV1(
                    num_tokens=self.num_tokens,
                    num_outputs=self.num_tokens + 1,  # +1 for F(s),
                    num_hid=args.cond_num_hidden,
                    num_layers=args.cond_num_layers,
                    max_len=self.max_len,
                    dropout=0,
                    arch=args.cond_arch_ver
                    )
            elif args.mlp == 'v2':
                self.model = GFNConditionalMLPV2(
                    num_tokens=self.num_tokens,
                    num_outputs=self.num_tokens + 1,  # +1 for F(s),
                    num_hid=args.cond_num_hidden,
                    num_layers=args.cond_num_layers,
                    max_len=self.max_len,
                    dropout=0,
                    arch=args.cond_arch_ver
                    )
            elif args.mlp == 'v3':
                self.model = GFNConditionalMLPV3(
                    num_tokens=self.num_tokens,
                    num_outputs=self.num_tokens + 1,  # +1 for F(s),
                    num_hid=args.cond_num_hidden,
                    num_layers=args.cond_num_layers,
                    max_len=self.max_len,
                    dropout=0,
                    arch=args.cond_arch_ver
                    )
            else:
                self.model = GFNConditionalMLP(
                    num_tokens=self.num_tokens,
                    num_outputs=self.num_tokens + 1,  # +1 for F(s),
                    num_hid=args.cond_num_hidden,
                    num_layers=args.cond_num_layers,
                    max_len=self.max_len,
                    dropout=0,
                    arch=args.cond_arch_ver
                    )
        self.model.to(args.device)
        self.target_model = copy.deepcopy(self.model)
        self.opt = torch.optim.Adam(self.model.model_params(), args.cond_lr, weight_decay=args.cond_L2,
                                    betas=(0.9, 0.999))

        self.device = args.device

        self.logsoftmax2 = torch.nn.LogSoftmax(2)

        self.fl = args.fl
        print("FL:",self.fl)
    def soft_update_params(self, tau):
        for param, target_param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    def train_step(self, strs, outcomes, actions, r):
        loss, info = self.get_loss(strs, outcomes, actions, r)
        loss.backward()
        if self.args.cond_clip > 0.:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.cond_clip)
        self.opt.step()
        self.opt.zero_grad()
        return loss, info

    def get_loss(self, strs, outcomes, actions, r):
        # REVIEW
        r = torch.tensor(r).to(self.device)
        #print(r)
        kl_loss = nn.KLDivLoss(reduction="batchmean")  # kl
        actions = torch.tensor(actions).to(self.device)
        ll_diff = []
        t_ll_diff = []

        outcomes = torch.stack(outcomes, dim=0).unsqueeze(1).repeat(1, self.max_len+1, 1)
     
        inp = torch.zeros(len(strs), self.max_len+1, self.max_len, self.num_tokens)
        for i, item in enumerate(strs):
            for j, k in enumerate(item):
               
                inp_item = F.one_hot(torch.tensor(k).to(self.device).long(), num_classes=self.num_tokens + 1)[:, :-1].to(torch.float32)
                inp[i, j, :len(k), :] = inp_item
       
        outcome_inp = F.one_hot(outcomes, num_classes=self.num_tokens + 1)[:, :, :, :-1].to(torch.float32)
        
        x = inp.reshape(inp.shape[0], self.max_len+1, -1).to(self.device).detach()
        outcome_x = outcome_inp.reshape(outcome_inp.shape[0], self.max_len+1, -1).to(self.device).detach()

        # s: max_len, N
        if self.fl:
            model_outs, p_b = self.model.forward_for_fl(x, outcome_x, None, return_all=False)  # max_len + 1, batch_size, num_outputs
        else:
            model_outs, p_b = self.model(x[:,:-1], outcome_x[:,:-1], None, return_all=False)  # max_len, batch_size, num_outputs
        pol_logits = model_outs[:, :, :-1]  # max_len (+1 if fl), N, action_dim
       
        if self.fl:
            pol_logits = pol_logits[:, :-1, :]
        log_flows = model_outs[:, :, -1]  # max_len (+1 if fl), N
        
        edge_mask = torch.zeros(*pol_logits.shape, device=self.device)
        
        for k, item in enumerate(strs):
            for j, it in enumerate(item[:-1]):
                if len(set(it)) < 1:
                    edge_mask[k, j, self.num_tokens//2:] = 1
                elif len(set(it)) == 1:
                    edge_mask[k, j, it[0] + self.num_tokens//2] = 1
        pol_logits = pol_logits - 1e7 * edge_mask#.log_softmax(1)
        pol_logits = self.logsoftmax2(pol_logits)  # batch_size, max_len, num_actions

        pol_logits = pol_logits.gather(2, actions.unsqueeze(-1)).squeeze(-1)



        if self.fl:

            fl_r = r.unsqueeze(1).repeat(1, pol_logits.shape[1])
            fl_cof = torch.ones(pol_logits.shape[1])
            for i in reversed(range(pol_logits.shape[1]-1)):
                fl_cof[i] = fl_cof[i+1]/2
            fl_cof = fl_cof.unsqueeze(0).repeat(pol_logits.shape[0], 1).to(pol_logits.device)
            fl_r = (fl_r*args.beta*fl_cof+ 1e-9).log()

        ll_diff = torch.zeros((pol_logits.shape)).to(self.device)
       
        if self.fl:
            ll_diff += log_flows[:, :-1]
        else:
            ll_diff += log_flows
        ll_diff += pol_logits
        # backward side
        if not self.fl:
            log_flows = log_flows[:, 1:]
            r = (r + 1e-9).log()
            r = r.unsqueeze(-1)
            end_log_flow = torch.cat((log_flows, r), -1)
            ll_diff -= end_log_flow
        else:
            log_flows = log_flows[:, :-1]

            log_flows = log_flows[:, 1:]
            r = r.to(dtype=torch.double)
            r = (self.reward_scale*r + 1e-9).log()
            
            r = r.unsqueeze(-1)
            end_log_flow = torch.cat((log_flows, r), -1)
            ll_diff -= end_log_flow

        back_edge_mask = torch.zeros(*p_b.shape, device=self.device)
        for k, item in enumerate(strs):
            for j, it in enumerate(item[:-1]):
                if len(set(it)) <= 1:
                    back_edge_mask[k, j, 0] = 1
        p_b = p_b - 1e7 * back_edge_mask
       
        back_logits = self.logsoftmax2(p_b)
        choice = (actions < (self.num_tokens // 2)).long().unsqueeze(-1)
       
        if self.fl:
            back_logits = back_logits[:, 1:].gather(2, choice).squeeze(-1)
        else:
            back_logits = back_logits[:, 1:].gather(2, choice[:, :-1]).squeeze(-1)
        ll_diff -= back_logits
        
        mask = torch.ones_like(ll_diff)
        loss = (ll_diff ** 2).sum() / mask.sum()
        error = None
        assert torch.isnan(loss).sum() == 0, print(loss)
        return loss, error

    def forward(self, x, outcome, return_all=False, coef=1, lens=None):
        assert not return_all

        x = torch.tensor(x, dtype=torch.long)

        inp_x = F.one_hot(x, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)
        outcome_inp_x = F.one_hot(outcome, num_classes=self.num_tokens + 1)[:, :, :-1].to(torch.float32)

        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        outcome_inp = torch.zeros(outcome.shape[0], self.max_len, self.num_tokens)
        assert outcome_inp.shape == outcome_inp_x.shape

        inp[:, :inp_x.shape[1], :] = inp_x
        outcome_inp[:, :outcome_inp_x.shape[1], :] = outcome_inp_x

        inp = inp.reshape(x.shape[0], -1).to(self.device)
        outcome_inp = outcome_inp.reshape(outcome.shape[0], -1).to(self.device)

        out = self.model(inp, outcome_inp, None, lens=lens, return_all=return_all) * coef
        return out

    def save(self, filename):
        # save model
        torch.save(self.model.state_dict(), filename + "_model")
        torch.save(self.opt.state_dict(), filename + "_optimizer")

    def load(self, filename):
        print('\033[32mload conditional_db from {}\033[0m'.format(filename))
        self.model.load_state_dict(torch.load(filename + "_model"))
     


class RolloutWorker:
    def __init__(self, args, oracle):
        self.oracle = oracle

        self.max_len = args.seq_max_len
        self.max_episodes_steps = int(np.ceil(self.max_len / args.num_bits))
        print('max_len: {}, num_bits: {}, max_episodes_steps=max_len/num_bits: {}'.format(self.max_len, args.num_bits,
                                                                                          self.max_episodes_steps))
        self.episodes_per_step = args.gen_episodes_per_step
        print('episodes_per_step: {}'.format(self.episodes_per_step))

        self.random_action_prob = args.gen_random_action_prob
        self.reward_exp = args.gen_reward_exp
        self.sampling_temperature = args.gen_sampling_temperature
        print('random_action_prob: {}, reward_exp: {}, sampling_temperature: {}'.format(self.random_action_prob,
                                                                                        self.reward_exp,
                                                                                        self.sampling_temperature))

        self.out_coef = args.gen_output_coef

        self.nbits = args.num_bits

        self.ints2s = lambda x: ''.join(f'{i:0{self.nbits}b}' for i in x)
        self.l2r = lambda x: x * self.reward_exp

        self.algo = args.algo
        self.num_tokens = 2 ** args.num_bits *2
        self.tokens = 2 ** args.num_bits
        self.cond_output_coef = args.cond_output_coef
        self.cond_random_action_prob = args.cond_random_action_prob
        self.cond_sampling_temperature = args.cond_sampling_temperature
    def execute_test_batch(self, model, device, use_rand_policy=False, use_tempered_policy=False, is_amortize=False, is_tabular=False, eval_num=None):
        episodes_per_step = eval_num if eval_num is not None else self.episodes_per_step

        # run an episode
        visited = []
        
        lists = lambda n: [list() for i in range(n)]

        states = lists(episodes_per_step)
        
        traj_states = [[[]] for i in range(episodes_per_step)]
        traj_actions = lists(episodes_per_step)
        traj_rewards = lists(episodes_per_step)
        traj_dones = lists(episodes_per_step)

        traj_logprob = np.zeros(episodes_per_step)
        
        bulk_trajs = []

        for t in (range(self.max_episodes_steps) if episodes_per_step > 0 else []):
            x = [states[i] for i in range(episodes_per_step)]

            if self.algo == 'sac':
                actions = model.policy.act(x, None)
            else:
                if is_tabular:
                    s_indicies = []
                    for s in states:
                        curr_s_idx = model.s_idx_map[str(s)]
                        s_indicies.append(curr_s_idx)
                    probs = model.all_real_pis[s_indicies]
                    # actions = probs.multinomial(1)

                    if use_tempered_policy and self.sampling_temperature != 1.:
                        cat = Categorical(logits=probs.log() / self.sampling_temperature)
                    else:
                        cat = Categorical(logits=probs.log())
                else:
                    with torch.no_grad():
                        logits, p_b = model(x, coef=self.out_coef)
                    if self.algo == 'db' and not is_amortize:
                        logits = logits[:, :-1] # the last dimension is for F(s)

                    if use_tempered_policy and self.sampling_temperature != 1.:
                        cat = Categorical(logits=logits / self.sampling_temperature)
                    else:
                        cat = Categorical(logits=logits)

                actions = cat.sample()

            if use_rand_policy and self.random_action_prob > 0:
                for i in range(actions.shape[0]):
                    if np.random.uniform(0, 1) < self.random_action_prob:
                        actions[i] = torch.tensor(np.random.randint(0, logits.shape[1])).to(device)

            if self.algo != 'sac':
                log_prob = cat.log_prob(actions)

            # Append predicted characters for active trajectories
            for (i, a) in enumerate(actions.cpu()):
                a = a.item()
                if self.algo != 'sac':
                    traj_logprob[i] += log_prob[i].item()
                if t == self.max_episodes_steps - 1:
                    final_s = states[i] + [a]
                    r_pre = self.oracle(self.ints2s(final_s))
                    r = self.l2r(r_pre)
                    d = 1
                    visited.append((final_s, r.item(), r_pre.item(), traj_logprob[i]))
                    bulk_trajs.append((final_s, r.item()))
                else:
                    r_pre = 0
                    r = 0
                    d = 0
                traj_states[i].append(states[i] + [a])
                traj_actions[i].append(a)
                traj_rewards[i].append(r)
                traj_dones[i].append(d)
                states[i] += [a]


        return {
            "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                "traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                "states": states,
                "bulk_trajs": bulk_trajs
            }
        }
    def execute_train_episode_batch(self, model, device):
        # run an episode
        visited = []
    
        lists = lambda n: [list() for i in range(n)]
    
        states = lists(self.episodes_per_step)
    
        traj_states = [[[]] for i in range(self.episodes_per_step)]
        traj_actions = lists(self.episodes_per_step)
        traj_rewards = lists(self.episodes_per_step)
        traj_dones = lists(self.episodes_per_step)
    
        traj_logprob = np.zeros(self.episodes_per_step)
    
        bulk_trajs = []
    
        for t in (range(self.max_episodes_steps) if self.episodes_per_step > 0 else []):
            x = [states[i] for i in range(self.episodes_per_step)]
    
            with torch.no_grad():
                logits, p_b = model(x, coef=self.out_coef)
            if self.algo == 'db':
                logits = logits[:, :-1]  # the last dimension is for F(s)
    
            cat = Categorical(logits=logits / self.sampling_temperature)
            actions = cat.sample()
    
            if self.random_action_prob > 0:
                for i in range(actions.shape[0]):
                    if np.random.uniform(0, 1) < self.random_action_prob:
                        actions[i] = torch.tensor(np.random.randint(0, logits.shape[1])).to(device)
    
            log_prob = cat.log_prob(actions)
    
            # append predicted characters for active trajectories
            for (i, a) in enumerate(actions.cpu()):
                a = a.item()
                traj_logprob[i] += log_prob[i].item()
                if t == self.max_episodes_steps - 1:
                    final_s = states[i] + [a]
                    r_pre = self.oracle(self.ints2s(final_s))
                    r = self.l2r(r_pre)
                    d = 1
                    visited.append((final_s, r.item(), r_pre.item(), traj_logprob[i]))
                    bulk_trajs.append((final_s, r.item()))
                else:
                    r_pre = 0
                    r = 0
                    d = 0
                traj_states[i].append(states[i] + [a])
                traj_actions[i].append(a)
                traj_rewards[i].append(r)
                traj_dones[i].append(d)
                states[i] += [a]
    
        return {
            "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                "traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                "states": states,
                "bulk_trajs": bulk_trajs
            }
        }

    def conditional_execute_train_episode_batch(self, model, device, outcomes_for_sample=None):
        # run an episode
        visited = []

        lists = lambda n: [list() for i in range(n)]

        states = lists(self.episodes_per_step)

        traj_states = [[[]] for i in range(self.episodes_per_step)]
        traj_actions = lists(self.episodes_per_step)
        traj_rewards = lists(self.episodes_per_step)
        traj_dones = lists(self.episodes_per_step)

        traj_logprob = np.zeros(self.episodes_per_step)

        bulk_trajs = []

        for t in (range(self.max_episodes_steps) if self.episodes_per_step > 0 else []):
            x = [states[i] for i in range(self.episodes_per_step)]

            y = [outcomes_for_sample[i] for i in range(self.episodes_per_step)]
            y = torch.stack(y)

            with torch.no_grad():
                if outcomes_for_sample is not None:
                    logits, p_b = model(x, y, coef=self.cond_output_coef)
                else:
                    logits, p_b = model(x, coef=self.out_coef)
            if args.algo == 'db':
               
                logits = logits[:, :-1]  # the last dimension is for F(s)
            edge_mask = torch.zeros((self.episodes_per_step, self.num_tokens))
            for k, item in enumerate(x):
                if len(set(item)) < 1:
                    edge_mask[k, self.tokens:] = 1
                elif len(set(item)) == 1:
                    edge_mask[k, item[0]+self.tokens] = 1
          
            logits = (logits - 1e7 * edge_mask.to(logits.device))#.log_softmax(1)
            if outcomes_for_sample is not None:
                sampling_temperature = self.cond_sampling_temperature
                random_action_prob = self.cond_random_action_prob
            else:
                sampling_temperature = self.sampling_temperature
                random_action_prob = self.random_action_prob
            sample_ins_probs = (logits / sampling_temperature).softmax(1)
            acts = sample_ins_probs.multinomial(1)
            actions = acts.squeeze(-1)
         
            for (i, a) in enumerate(actions.cpu()):
                a = a.item() #REVIEW
                act = a

                #append=True
                if a < self.tokens:
                    append = True
                else:
                    a = a - self.tokens
                    append = False
               
                if t == self.max_episodes_steps - 1:
                    final_s = states[i] + [a] if append else [a] + states[i]
                  
                    r = 1.
                    d = 1
              
                    bulk_trajs.append((final_s, r))
                else:
                    r_pre = 0
                    r = 0
                    d = 0
                states[i] = states[i] + [a] if append else [a] + states[i]
                traj_states[i].append(states[i].copy())
                traj_actions[i].append(copy.copy(act))
                traj_rewards[i].append(r)
                traj_dones[i].append(d)

        return {
            "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                "traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                "states": states,
                "bulk_trajs": bulk_trajs
            }
        }
    def conditional_execute_backward(self, model, device, outcomes_for_sample=None):
        # run an episode
        visited = []

        lists = lambda n: [list() for i in range(n)]

        states = outcomes_for_sample.cpu().numpy().tolist()#lists(self.episodes_per_step)

        traj_states = [[states[i].copy()] for i in range(self.episodes_per_step)]
        traj_actions = lists(self.episodes_per_step)
        traj_rewards = lists(self.episodes_per_step)
        traj_dones = lists(self.episodes_per_step)

        traj_logprob = np.zeros(self.episodes_per_step)

        bulk_trajs = []

        for t in (range(self.max_episodes_steps) if self.episodes_per_step > 0 else []):
            x = [states[i] for i in range(self.episodes_per_step)]

            y = [outcomes_for_sample[i] for i in range(self.episodes_per_step)]
            y = torch.stack(y)

            with torch.no_grad():
                if outcomes_for_sample is not None:
                    logits, p_b = model(x, y, coef=self.cond_output_coef)
                else:
                    logits, p_b = model(x, coef=self.out_coef)
            if args.algo == 'db':
                logits = logits[:, :-1]  # the last dimension is for F(s)

         
            edge_mask = torch.zeros((self.episodes_per_step, 2))
            for k, item in enumerate(x):
                if len(set(item)) <= 1:
                    edge_mask[k, 0] = 1
            p_b = (p_b - 1e7 * edge_mask.to(logits.device)).log_softmax(1)


            pb_probs = p_b.softmax(1) #REVIEW
            pb_acts = pb_probs.multinomial(1).squeeze(-1)
    
            # append predicted characters for active trajectories
            for (i, a) in enumerate(pb_acts.cpu()):
              

                if t == 0:
                    final_s = states[i]
                 
                    r =1.
                    d = 1
               
                    bulk_trajs.append((final_s, r))
                else:
                    r_pre = 0
                    r = 0
                    d = 0
                act = states[i].pop(-1) if a > 0 else states[i].pop(0)+self.tokens
              
                traj_states[i].append(states[i].copy())
                traj_actions[i].append(copy.copy(act))
            
                traj_dones[i].append(d)

        return {
            "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                #"traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                "states": states,
                "bulk_trajs": bulk_trajs
            }
        }

class TestOracle:
    def __init__(self, args, to_print=False):
        self.vocab = list('01')
        self.nbits = args.num_bits
        self.oracle_difficulty = args.oracle_difficulty

        self.sp_r = args.sp_r
        self.sp_r_thres = args.sp_r_thres

        if self.oracle_difficulty == "easy":
            # self.modes = [[0] * args.mode_max_len for _ in range(args.mode_max_len)]
            # for i in range(len(self.modes)):
            #     self.modes[i][i] = 1

            # self.modes = ["".join([str(b) for b in s]) for s in self.modes]
            # self.modes = random.sample(self.modes, args.num_test_modes)
            self.modes = [[0] * args.seq_max_len for _ in range(args.mode_max_len)]
            for i in range(len(self.modes)):
                self.modes[i][i] = 1

            self.modes = ["".join([str(b) for b in s]) for s in self.modes]
            self.modes = random.sample(self.modes, args.seq_max_len)

        elif self.oracle_difficulty == "medium":
            if args.num_bits == 2 and args.seq_max_len == 40:
                self.modes = ['0110111111100100000101100111101010110111', '0010111110101111010011010100000110001101',
                              '1001011111101100100110100100011010000011', '0101111101111011001000011001011110001011',
                              '1101001011001010101010001010100000100101', '0010010100110001100000101000111001111001',
                              '0110100101111000111011110011000110111111', '0001010110001001111010000111010011111101',
                              '0111111111100001110111111100001101001011', '0100001110101000011101001110110101100111',
                              '0111111001010011111011100110111000100001', '0011100111010101110100101001000000000101',
                              '1101110011110101000011111010101011000111', '1001001000101010110010101000011010110101',
                              '0000110000110010111111111110011110010011', '0101001000001010011110010111100100111001',
                              '0110100011010100100000111111110000111101', '1001110111011111001000101010110100010101',
                              '1101101111100111101001010110110100001001', '1000101000011100110110110101100001100111',
                              '0001010010001100010010100110011100000101', '0000111110011111111000111100101111100001',
                              '1000011111010011111001001110011000110101', '1011110001111000000110100101000010011001',
                              '1010101110101100011000000100001000110101', '0010001110010110000000010101000101100011',
                              '0111010000010001101010110101001000101001', '0100111010111110010100110001100000011111',
                              '1111000000010101011011010001100100101101', '0111001001100001111001001000101101100001',
                              '0111011010000111110101111100110011110001', '0001100001100101100110100110010001110111',
                              '0011110110011111001010110111111001100001', '1100001000111011010110011010001111101001',
                              '1010001111011001010101011111101010101011', '0001011110101011011111111010100011011011',
                              '0010111101110011111010010100101111110011', '0010110111101010010110000101001010011111',
                              '1110100100101001001100101101101000100001', '1101110111110110000111110100011000000101',
                              '0000101001011111101001110110101010110111', '0010110010101111111000100001101111101011',
                              '0010100101010011010001110001011111000011', '1100100010101001110101000101000010010111',
                              '0011111111010011001111000000011110010001', '0111101010100000000011111011011011000011',
                              '1101010110001100001001101011010011011101', '1101000010001000000000000110111011011111',
                              '0101111100010100010101111111111010010111', '0100001110100000011100111011001100001111',
                              '1100101111011101001001101011100010110101', '1011001011111000100010001011010100101011',
                              '1111001001111001001001000110101010010011', '1000110000100111100110001110001100111001',
                              '1010001010111110011101010100111100000011', '0001010010011001111001011011001000000101',
                              '1110011110101001111000001010010010101001', '1100000110110001011111111101111000011101',
                              '1001001100100010001010111101011000110111', '0000110111111011101111101100111010100011']
            elif args.num_bits == 3 and args.seq_max_len == 60:
                self.modes = ['011011111110010000010110011110101011011001011111010111101001',
                              '110101000001100011010010111111011001001101001000110100000101',
                              '101111101111011001000011001011110001011101001011001010101011',
                              '000101010000010010001001010011000110000010100011100111100011',
                              '101001011110001110111100110001101111100010101100010011110101',
                              '000111010011111100111111111100001110111111100001101001010101',
                              '000111010100001110100111011010110011011111100101001111101111',
                              '001101110001000000111001110101011101001010010000000001011011',
                              '110011110101000011111010101011000111001001000101010110010101',
                              '100001101011010000011000011001011111111111001111001001010101',
                              '010000010100111100101111001001110001101000110101001000001111',
                              '111110000111101001110111011111001000101010110100010101101101',
                              '111110011110100101011011010000100100010100001110011011011011',
                              '011000011001100010100100011000100101001100111000001000001111',
                              '110011111111000111100101111100001000011111010011111001001111',
                              '001100011010101111000111100000011010010100001001100101010111',
                              '101011000110000001000010001101000100011100101100000000101011',
                              '000101100010111010000010001101010110101001000101000100111011',
                              '011111001010011000110000001111111100000001010101101101000111',
                              '001001011001110010011000011110010010001011011000001110110101',
                              '000111110101111100110011110000001100001100101100110100110011',
                              '000111011001111011001111100101011011111100110000110000100011',
                              '110110101100110100011111010010100011110110010101010111111011',
                              '010101010001011110101011011111111010100011011010010111101111',
                              '001111101001010010111111001001011011110101001011000010100101',
                              '100111111101001001010010011001011011010001000011011101111101',
                              '110000111110100011000000100000101001011111101001110110101011',
                              '011011001011001010111111100010000110111110101001010010101001',
                              '110100011100010111110000111001000101010011101010001010000101',
                              '010110011111111010011001111000000011110010000111101010100001',
                              '000001111101101101100001110101011000110000100110101101001101',
                              '111011010000100010000000000001101110110111101011111000101001',
                              '010101111111111010010110100001110100000011100111011001100001',
                              '111110010111101110100100110101110001011010101100101111100011',
                              '000100010110101001010111110010011110010010010001101010100101',
                              '011000110000100111100110001110001100111001010001010111110011',
                              '110101010011110000001000101001001100111100101101100100000011',
                              '011100111101010011110000010100100101010011000001101100010111',
                              '111111101111000011101001001100100010001010111101011000110111',
                              '000011011111101110111110110011101010001111101110111011010111',
                              '001111101011101100111100100000110100000111111101101010101011',
                              '100111000000101000101100011000110001110101011010100111001011',
                              '001000100101101111100011011101111111101100011100010110101011',
                              '010011010001001101101000000001010100010101101000000001101111',
                              '011100100100011011001011100011110011001111101111101100100111',
                              '011010100010111101010111111110111000011101101011001011010101',
                              '010001100000110100011100011110100111110011010000101111110001',
                              '101001011101100010100111010011110011000110101011000010100011',
                              '110011100101110011011110110101010110001100100000000100111101',
                              '011101100101101110001011110111101111100010111011011110011001',
                              '100100000011100101011111100010101101110101111110011110001001',
                              '100010110000001110001101110101110000001010000010100110101111',
                              '101111111000101101101011010100101110011111011011011100110011',
                              '001001110011011111001100001111110111100001100001101000101111',
                              '101100011011010001111001001101010010011110000010010111011101',
                              '011000001101001001010111011100110000100000101101010000011111',
                              '000001011110011011011111000000100000001000011100000001111011',
                              '111101111000111000010010000001111111000000010101010100010011',
                              '110101111010011100111011010110101001011110111001111100110001',
                              '101110100000101111110100000110000100010101010011000111101101']
            elif args.num_bits == 4 and args.seq_max_len == 80:
                self.modes = ['01101111111001000001011001111010101101100101111101011110100110101000001100011011',
                              '00101111110110010011010010001101000001010111110111101100100001100101111000101111',
                              '01001011001010101010001010100000100100010010100110001100000101000111001111000111',
                              '01001011110001110111100110001101111100010101100010011110100001110100111111001111',
                              '11111110000111011111110000110100101010000111010100001110100111011010110011011111',
                              '11001010011111011100110111000100000011100111010101110100101001000000000101101111',
                              '00111101010000111110101010110001110010010001010101100101010000110101101000001101',
                              '00011001011111111111001111001001010100100000101001111001011110010011100011010001',
                              '11010100100000111111110000111101001110111011111001000101010110100010101101101111',
                              '11001111010010101101101000010010001010000111001101101101011000011001100010100101',
                              '00110001001010011001110000010000011111001111111100011110010111110000100001111101',
                              '10011111001001110011000110101011110001111000000110100101000010011001010101110101',
                              '11000110000001000010001101000100011100101100000000101010001011000101110100000101',
                              '00110101011010100100010100010011101011111001010011000110000001111111100000001011',
                              '01011011010001100100101100111001001100001111001001000101101100000111011010000111',
                              '11101011111001100111100000011000011001011001101001100100011101100111101100111111',
                              '00101011011111100110000110000100011101101011001101000111110100101000111101100101',
                              '10101011111101010101010001011110101011011111111010100011011010010111101110011111',
                              '10100101001011111100100101101111010100101100001010010100111111101001001010010011',
                              '10010110110100010000110111011111011000011111010001100000010000010100101111110101',
                              '01110110101010110110010110010101111111000100001101111101010010100101010011010001',
                              '11100010111110000111001000101010011101010001010000100101100111111110100110011111',
                              '00000001111001000011110101010000000001111101101101100001110101011000110000100111',
                              '01011010011011101101000010001000000000000110111011011110101111100010100010101111',
                              '11111110100101101000011101000000111001110110011000011111001011110111010010011011',
                              '01110001011010101100101111100010001000101101010010101111100100111100100100100011',
                              '10101010010011000110000100111100110001110001100111001010001010111110011101010101',
                              '01111000000100010100100110011110010110110010000001011100111101010011110000010101',
                              '01001010100110000011011000101111111110111100001110100100110010001000101011110101',
                              '11000110110000110111111011101111101100111010100011111011101110110101100111110101',
                              '11101100111100100000110100000111111101101010101011001110000001010001011000110001',
                              '11000111010101101010011100101001000100101101111100011011101111111101100011100011',
                              '01101010101001101000100110110100000000101010001010110100000000110111011100100101',
                              '00110110010111000111100110011111011111011001001101101010001011110101011111111011',
                              '11000011101101011001011010100100011000001101000111000111101001111100110100001011',
                              '11111000101001011101100010100111010011110011000110101011000010100011100111001011',
                              '11001101111011010101011000110010000000010011110011101100101101110001011110111101',
                              '11111000101110110111100110010010000001110010101111110001010110111010111111001111',
                              '10001001000101100000011100011011101011100000010100000101001101011110111111100011',
                              '01101101011010100101110011111011011011100110010010011100110111110011000011111101',
                              '11110000110000110100010111101100011011010001111001001101010010011110000010010111',
                              '10111001100000110100100101011101110011000010000010110101000001111000001011110011',
                              '10110111110000001000000010000111000000011110111110111100011100001001000000111111',
                              '11000000010101010100010011101011110100111001110110101101010010111101110011111001',
                              '11000101110100000101111110100000110000100010101010011000111101101010000001001111',
                              '11010010010111100101001100001000110010001000110000011010110111101001000001001111',
                              '11000000111001110111101111100011110010101100001010110110111110010001101000010001',
                              '01101111110110010110110010011111010100001110001100011001101001011111000011010101',
                              '10000101100101001110101010101110010001111001111111110110010010011100001100100001',
                              '10011001010010100110100010101110110110110010000001001011111011001001100011000111',
                              '11001100010000011111000011111000100110101110100100111110100000100101111000101001',
                              '01000100101110111110101111111000111111010001110011000000100101010000111111000001',
                              '10001111010110100100011111000000110110111100011000010100110011010100101000000001',
                              '00110010000001110000001110111111000101100111011111101100010001111000101011100111',
                              '00111101110011100101011101110101100111000100110101000011111001011101111110110111',
                              '01111100011101011110111111100000101100011011110001111100011011111100101111111101',
                              '00010000110110101010001011111100011101101101011010110110011101100000111010110101',
                              '01011011011100101001110010010000001100110110110000010011000100010000111010000001',
                              '00010110111100000010010111000010001110110111001011001001101101100100000011001011',
                              '01000101000010001101000010111011010010111111000010111011000011001001100101101101']
            elif args.num_bits == 5 and args.seq_max_len == 100:
                self.modes = [
                    '0110111111100100000101100111101010110110010111110101111010011010100000110001101001011111101100100111',
                    '0100100011010000010101111101111011001000011001011110001011101001011001010101010001010100000100100011',
                    '0010100110001100000101000111001111000110100101111000111011110011000110111110001010110001001111010001',
                    '0111010011111100111111111100001110111111100001101001010100001110101000011101001110110101100110111111',
                    '1001010011111011100110111000100000011100111010101110100101001000000000101101110011110101000011111011',
                    '0101011000111001001000101010110010101000011010110100000110000110010111111111110011110010010101001001',
                    '0001010011110010111100100111000110100011010100100000111111110000111101001110111011111001000101010111',
                    '0100010101101101111100111101001010110110100001001000101000011100110110110101100001100110001010010001',
                    '1100010010100110011100000100000111110011111111000111100101111100001000011111010011111001001110011001',
                    '0110101011110001111000000110100101000010011001010101110101100011000000100001000110100010001110010111',
                    '0000000010101000101100010111010000010001101010110101001000101000100111010111110010100110001100000011',
                    '1111111000000010101011011010001100100101100111001001100001111001001000101101100000111011010000111111',
                    '0101111100110011110000001100001100101100110100110010001110110011110110011111001010110111111001100001',
                    '1100001000111011010110011010001111101001010001111011001010101011111101010101010001011110101011011111',
                    '1111010100011011010010111101110011111010010100101111110010010110111101010010110000101001010011111111',
                    '0100100101001001100101101101000100001101110111110110000111110100011000000100000101001011111101001111',
                    '0110101010110110010110010101111111000100001101111101010010100101010011010001110001011111000011100101',
                    '0010101001110101000101000010010110011111111010011001111000000011110010000111101010100000000011111011',
                    '1011011000011101010110001100001001101011010011011101101000010001000000000000110111011011110101111101',
                    '0010100010101111111111010010110100001110100000011100111011001100001111100101111011101001001101011101',
                    '0010110101011001011111000100010001011010100101011111001001111001001001000110101010010011000110000101',
                    '0111100110001110001100111001010001010111110011101010100111100000010001010010011001111001011011001001',
                    '0000101110011110101001111000001010010010101001100000110110001011111111101111000011101001001100100011',
                    '0001010111101011000110110000110111111011101111101100111010100011111011101110110101100111110101110111',
                    '0011110010000011010000011111110110101010101100111000000101000101100011000110001110101011010100111001',
                    '1010010001001011011111000110111011111111011000111000101101010101001101000100110110100000000101010001',
                    '1010110100000000110111011100100100011011001011100011110011001111101111101100100110110101000101111011',
                    '0101111111101110000111011010110010110101001000110000011010001110001111010011111001101000010111111001',
                    '0101001011101100010100111010011110011000110101011000010100011100111001011100110111101101010101100011',
                    '1001000000001001111001110110010110111000101111011110111110001011101101111001100100100000011100101011',
                    '1111100010101101110101111110011110001001000101100000011100011011101011100000010100000101001101011111',
                    '0111111100010110110101101010010111001111101101101110011001001001110011011111001100001111110111100001',
                    '1100001101000101111011000110110100011110010011010100100111100000100101110111001100000110100100101011',
                    '1101110011000010000010110101000001111000001011110011011011111000000100000001000011100000001111011111',
                    '1011110001110000100100000011111110000000101010101000100111010111101001110011101101011010100101111011',
                    '1100111110011000101110100000101111110100000110000100010101010011000111101101010000001001111101001001',
                    '1011110010100110000100011001000100011000001101011011110100100000100111110000001110011101111011111001',
                    '0111100101011000010101101101111100100011010000100001101111110110010110110010011111010100001110001101',
                    '0011001101001011111000011010101000010110010100111010101010111001000111100111111111011001001001110001',
                    '0110010000100110010100101001101000101011101101101100100000010010111110110010011000110001111001100011',
                    '0000011111000011111000100110101110100100111110100000100101111000101000100010010111011111010111111101',
                    '0011111101000111001100000010010101000011111100000100011110101101001000111110000001101101111000110001',
                    '0101001100110101001010000000000110010000001110000001110111111000101100111011111101100010001111000101',
                    '1011100110011110111001110010101110111010110011100010011010100001111100101110111111011011011111000111',
                    '1010111101111111000001011000110111100011111000110111111001011111111000010000110110101010001011111101',
                    '0011101101101011010110110011101100000111010110100101101101110010100111001001000000110011011011000001',
                    '1001100010001000011101000000000101101111000000100101110000100011101101110010110010011011011001000001',
                    '0110010101000101000010001101000010111011010010111111000010111011000011001001100101101101001010110001',
                    '0110111011100001011010001111011001001001011111110110011011110110000000100010100110000110000110000111',
                    '0010010011000000100100100011010100111110111100110000001110111000101110011110010101111001010101111101',
                    '0110111110101010001000000100101010010010000011111110110110011000000000110100010011110100110110110011',
                    '1010110011110011001100000111110000110011011111001101000010010010111100000111101100001010110011011111',
                    '0011001110010101110111001001110101100111101100100011101011100111001111101001111100100100001111101001',
                    '1111100010101011001000010101001001111110000000101010001001000001000011101100101111000011111101100101',
                    '1000000000111010000100101000001011111101000100111011100110010011001101101110111111110111111000101101',
                    '1000000011001001011110001111000011010111110110001100111110001111100110010110111001011010111101000011',
                    '1000001001100011010110010110111010010001001000100011101001000010000001000011101001000100010110111011',
                    '0110110111010011111011010100011100000010011111101100100100111100100011000010010000101010100000111101',
                    '0101001000101111100000100101100011101100101110011110000010111111110011110001110001011101101111011101',
                    '0111001101000100110000001001100110110100001001100101111111100100010101110100111000111010101101100001']

        elif self.oracle_difficulty == "hard":
            vocab = ['00000000', '11111111', '11110000', '00001111', '00111100']
            self.modes = ["".join(random.choices(vocab, k=random.randint((args.mode_max_len // len(vocab[0])) * 0.5,
                                                                         (args.mode_max_len // len(vocab[0]))))) for _
                          in range(args.num_test_modes)]

        # hardest
        elif self.oracle_difficulty == "hardest":
            randstr = lambda n: ''.join([f"{np.random.randint(2):b}" for i in range(n - 1)]) + '1'
            self.modes = [randstr(np.random.randint(args.seq_min_len, args.mode_max_len)) for i in
                          range(args.num_test_modes)]

        self.formatted_modes = []
        for mode_idx in range(len(self.modes)):
            curr_mode = self.modes[mode_idx]
            curr_mode_split = self.split_string(curr_mode, self.nbits)
            curr_mode_split_formatted = [int(cms, 2) for cms in curr_mode_split]
            if self.nbits <= 3:
                curr_mode_split_formatted_str = "".join([str(ch) for ch in curr_mode_split_formatted])
            else:
                self.alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
                                 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7',
                                 '8', '9']
                assert len(self.alphabet) >= 2 ** self.nbits

                curr_mode_split_formatted_str = "".join([self.alphabet[ch_idx] for ch_idx in curr_mode_split_formatted])
            self.formatted_modes.append(curr_mode_split_formatted_str)

        if to_print:
            print('modes_len', [len(self.modes[i]) for i in range(len(self.modes))])
            print('modes', self.modes)
            print('formatted_modes', self.formatted_modes)

        self.build_test_set()

    def split_string(self, string, length):
        return [string[i:i + length] for i in range(0, len(string), length)]

    def __call__(self, s, to_print=False, sp_r=False):
        s_split = self.split_string(s, self.nbits)
        s_split_formatted = [int(ss, 2) for ss in s_split]
        if self.nbits <= 3:
            s_split_formatted_str = "".join([str(ch) for ch in s_split_formatted])
        else:
            s_split_formatted_str = "".join([self.alphabet[ch_idx] for ch_idx in s_split_formatted])
        dists_new = [levenshtein(s_split_formatted_str, i) for i in self.formatted_modes]
        m_new = np.argmin(dists_new)

        if self.sp_r:
            if dists_new[m_new] <= self.sp_r_thres:
                reward_new = 1 - dists_new[m_new] / len(self.formatted_modes[m_new])
            else:
                reward_new = np.log(1e-6)
        else:
            reward_new = 1 - dists_new[m_new] / len(self.formatted_modes[m_new])

        reward = reward_new

        if to_print:
            print('s', s)
            print('s_rstrip', s_rstrip)
            print('s_split', s_split)
            print('s_split_formatted', s_split_formatted)
            print('s_split_formatted_str', s_split_formatted_str)
            print('dists', dists)
            print('dists_new', dists_new)
            print('m', m, 'm_new', m_new)
            print('min_dist', dists[m], 'min_dist_new', dists_new[m_new])
            print('reward', reward, 'reward_new', reward_new)

        return torch.tensor(reward)

    def compute_dists(self, s):
        s_split = self.split_string(s, self.nbits)
        s_split_formatted = [int(ss, 2) for ss in s_split]
        if self.nbits <= 3:
            s_split_formatted_str = "".join([str(ch) for ch in s_split_formatted])
        else:
            s_split_formatted_str = "".join([self.alphabet[ch_idx] for ch_idx in s_split_formatted])
        dists = np.array([levenshtein(s_split_formatted_str, i) for i in self.formatted_modes])
        return dists

    def build_test_set(self):
        self.test_seq = []
        self.test_rs = []

        def noise_seq(x, n):
            x = list(x)
            idces = list(range(len(x)))
            for i in range(n):
                j = idces.pop(np.random.randint(len(idces)))
                r = x[j]
                while r == x[j]:
                    r = self.vocab[np.random.randint(len(self.vocab))]
                x[j] = r
            return ''.join(x)

        for m in self.modes:
            for n in range(1, len(m) + 1):
                s = noise_seq(m, n)
                self.test_seq.append(s)
                self.test_rs.append(self(s))

        test_rs = torch.stack(self.test_rs).numpy().tolist()


class Logger:
    def __init__(self):
        self.data = {}

    def add(self, key, value):
        if key in self.data.keys():
            self.data[key].append(value)
        else:
            self.data[key] = [value]

    def update(self, key, value):
        if key in self.data.keys():
            self.data[key] = value
        else:
            self.data[key] = value

    def save(self, save_path, args):
        pickle.dump({
            'logged_data': self.data,
            'args': args},
            gzip.open(save_path, 'wb')
        )


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


def train_generators(args, generator, oracle, outdir):
    if args.wdb:
        wandb.init(project='gfn_bit', name='{}_{}'.format(args.exp_name, args.seed))

    print("Training generators (Unsupervised Stage)")

    visited = []
    uncond_losses = []

    test_nlls = []
    test_Zs = []
    regressions = []
    
    args.logger.add('modes', oracle.modes)
    for om_idx in range(len(oracle.modes)):
        print('oracle mode[{}]: {}, len={}'.format(om_idx, oracle.modes[om_idx], len(oracle.modes[om_idx])))

    ints2s = lambda x: ''.join(f'{i:0{args.num_bits}b}' for i in x)
    #generator.load('/home/hhebb/ti_gflownet/results_bit/20240305T222450.440282/models/step_1000')
    rollout_worker = RolloutWorker(args, oracle)
    train_mode_cnt_map = {}
    train_mode_cnts = []
    buffer = PrioritizedReplayBuffer(max_size=int(1e6), alpha=args.p_alpha, beta=args.p_beta, beta_steps=args.beta_steps, tl=tl)
    sub_len = 1
    import re
    for it in tqdm(range(args.gen_num_iterations + 1), disable=True):
        
     
        max_episodes_steps = int(np.ceil(args.seq_max_len / args.num_bits))
        outcomes_for_sample = torch.randint(low=0, high=2 ** args.num_bits,
                                            size=(args.gen_episodes_per_step, max_episodes_steps))

        outcomes = torch.LongTensor(outcomes_for_sample).to(args.device)

    
        cond_rollout_artifacts = rollout_worker.conditional_execute_train_episode_batch(generator, args.device,
                                                                                        outcomes_for_sample=outcomes)

        # process data sampled by unconditional gfn (gafn)
        cond_sampled_strs, _ = zip(*cond_rollout_artifacts["trajectories"]["bulk_trajs"])
        states = cond_rollout_artifacts["trajectories"]["traj_states"]
        actions = cond_rollout_artifacts["trajectories"]["traj_actions"]
       
        cond_sampled_s = torch.tensor(cond_sampled_strs,
                                      dtype=torch.long).to(args.device)  # .to(self.device) # batch_size, str_len=N/K (every row is a sequence)
        outcomes_minus = cond_sampled_s
      
        cond_converted_Rs = torch.prod(outcomes == outcomes_minus, -1).float()
       
        buffer.add(states, outcomes, cond_converted_Rs, actions)
        back_Rs = torch.ones(cond_converted_Rs.shape[0])
        back_cond_rollout_artifacts = rollout_worker.conditional_execute_backward(generator, args.device,
                                                                                        outcomes_for_sample=outcomes)

        b_states = back_cond_rollout_artifacts["trajectories"]["traj_states"]
        b_actions = back_cond_rollout_artifacts["trajectories"]["traj_actions"]

        for i in range(len(b_states)):
            b_states[i].reverse()
            b_actions[i].reverse()
      
        buffer.add(b_states, outcomes, back_Rs, b_actions, if_back=True)
        samples, weights, tree_idxs, length = buffer.sample(args.batch_size)
        cond_loss_neg, priority = generator.train_step(strs=samples[0], outcomes=samples[1], actions=samples[2],
                                                                 r=samples[3])
        buffer.update_priorities(tree_idxs, priority.detach().cpu().numpy())
      
        formatted_cond_converted_Rs = cond_converted_Rs.cpu().int().numpy().tolist()
        success_cnt, tmp_cnt = 0, 0
        for sample_idx in range(len(formatted_cond_converted_Rs)):
            curr_conditional_r = formatted_cond_converted_Rs[sample_idx]
          
            if curr_conditional_r == 1:
                print('\033[32m1\033[0m', end=', ')
                success_cnt += 1
            else:
                print('0', end=', ')
        success_cnt = success_cnt * 100.0 / len(formatted_cond_converted_Rs)
     
        print('-> {}'.format(success_cnt))
        record_stats(outdir, [it, success_cnt], 'conditional_success_rate')

        if args.wdb:
            wandb.log({
                'priority':priority.mean().item(),
                'success_cnt': success_cnt,
                # 'sub_success_cnt':tmp_cnt,
                'loss': cond_loss_neg.item()
            })
        if it % args.target_update_frequency == 0:
            generator.soft_update_params(args.tau)
        if args.save_model and (it + 1) % 1000 == 0:
            generator.save('{}/models/step_{}'.format(outdir, it + 1))
       
def rollout(model, episodes, curr_s=None, outcomes_for_sample=None, device=None, max_len=8, num_tokens=4, exp_weight=0., temp=1.):
    states = curr_s if curr_s is not None else [[] for _ in range(episodes)]

    traj_logprob = torch.zeros(episodes).to(device)

    for t in (range(max_len - len(curr_s[0])) if episodes > 0 else []):
        x = torch.tensor(states, dtype=torch.long).to(device) # batch_size, str_len=N/K (every row is a sequence)

        # format the input
        inp_x = F.one_hot(x, num_classes=num_tokens + 1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], max_len, num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        inp = inp.reshape(x.shape[0], -1).to(device)

        logits, p_b = model(inp, outcomes_for_sample, None) # batch_size, |A|
        
       
        edge_mask = torch.zeros_like(logits).to(device)
        sample_ins_probs = (1 - exp_weight) * (logits / temp).softmax(1) + exp_weight * (1 - edge_mask) / (1 - edge_mask + 0.0000001).sum(1).unsqueeze(1)
        cat = Categorical(probs=sample_ins_probs)
       
        
        actions = cat.sample()

        traj_logprob += cat.log_prob(actions)
        
      
        for i, a in enumerate(actions):
            states[i] += [a.item()]
    return states, traj_logprob

def main(args):
    torch.set_num_threads(1)

    outdir = prepare_output_dir(args.dir, argv=sys.argv)
    print('\033[32moutdir: {}\033[0m'.format(outdir))

    # args.device = torch.device('cuda')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.logger = Logger()

    test_oracle = TestOracle(args)

    generator = ConditionalDBGFlowNetGenerator(args)

    train_generators(args, generator, test_oracle, outdir)


_base = {
    'gen_do_pg': 0,
    'gen_num_iterations': 50000,
    'gen_episodes_per_step': 16,
    'gen_data_sample_per_step': 0,
    'gen_reward_norm': 4,
    'gen_reward_exp': 1,
    'gen_L2': 1e-5,
    'gen_loss_eps': 1e-10,
    'gen_learning_rate': 1e-4,
}

_array = [{**_base, 'num_bits': n, 'seed': s} for n in range(1, 17) for s in range(4)]

for i, a in enumerate(_array):
    a['save_path'] = f'results/test/nbits_comp_{i}.pkl.gz'

if __name__ == '__main__':
    args = parser.parse_args()
    tf = lambda x: torch.FloatTensor(x).to(args.device)
    tl = lambda x: torch.LongTensor(x).to(args.device)
    args.mode_max_len = args.seq_max_len
   
    assert args.cond_model_type == args.gen_model_type
    main(args)
    
