#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%pdb
"""
Code for the paper "Learning to Schedule Joint Radar-Communication with Deep
Multi-Agent Reinforcement Learning", IEEE Transactions on Vehicular Technology

Author: Joash Lee

This program uses the Multi-Agent Proximal Policy Optimization (MAPPO) algorithm
to solve the Joint Radar-Communication (JRC) and Age of Information (AoI) Markov
Game in the file 'JRCwithAOI_multi.py'

"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gym
import logger
import os
import time
import inspect
from torch.distributions import Categorical, Normal
from typing import Callable, Union

from JRCwithAOI_multi import AV_Environment
import json


# In[]

# device = torch.device(0 if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)

def save_itr_info(fname, itr, av_reward):
    with open(fname, "w") as f:
        f.write(f"Iteration: {itr}\n Average reward: {av_reward}\n")

def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logger.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(train_AC)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logger.save_params(params)
    
def save_variables(model, model_file):
    """Save parameters of the NN 'model' to the file destination 'model_file'. """
    torch.save(model.state_dict(), model_file)

def load_variables(model, load_path):
#    model.cpu()
    model.load_state_dict(torch.load(load_path)) #, map_location=lambda storage, loc: storage))
    # model.eval() # TODO1- comment:  to set dropout and batch normalization layers to evaluation mode before running inference
    
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def constant_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return initial_value

    return func
    
    
# In[]
    
class Network(nn.Module):
    def __init__(self, ob_dim, policy_out_dim, policy_size, vf_size, discrete):
        super(Network, self).__init__()
        self.out_dim = policy_out_dim
        self.discrete = discrete
        
        last_layer_dim_pi = last_layer_dim_vf = ob_dim
        
        # Input layer
        policy_net, value_net = [], []
        for layer_size in policy_size:
            policy_net.append(nn.Linear(last_layer_dim_pi, layer_size))
            policy_net.append(nn.ReLU())
            last_layer_dim_pi = layer_size
        policy_net.append(nn.Linear(last_layer_dim_pi, policy_out_dim))
        
        for layer_size in vf_size:
            value_net.append(nn.Linear(last_layer_dim_vf, layer_size))
            value_net.append(nn.ReLU())
            last_layer_dim_vf = layer_size
        value_net.append(nn.Linear(last_layer_dim_vf, 1))
            
        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)
        
    def forward(self, ob):
        action_logits = self.policy_net(ob)
        value = self.value_net(ob)
        
        if self.discrete == False:
            mean, logstd = torch.split(action_logits, int(self.out_dim/2), dim = int(action_logits.dim()-1))
            dist = Normal(mean, logstd.exp())
        else:
            dist = Categorical(logits = action_logits)
        
        return dist, value

class CNN(nn.Module):
    def __init__(self, ob_dim, policy_out_dim, policy_size, vf_size, discrete):
        super(CNN, self).__init__()
        self.out_dim = policy_out_dim
        self.discrete = discrete
        
        # 2 x 11 x 1 --> 3 x 12 x 10
        self.conv1p = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=2, padding = 1)
        self.conv1v = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=2, padding = 1)
        
        # 3 x 12 x 10 --> 1 x 5 x 10
        self.pool1p = nn.MaxPool2d(kernel_size=3, stride=2)
        self.pool1v = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.linear1p = nn.Linear(12, policy_size[0])
        self.linear1v = nn.Linear(12, policy_size[0])
        
        # 1 x 5 x 10 = 50.  Input size  =50 + policy_size[0]
        self.linear2p = nn.Linear(50 + policy_size[0], policy_size[1]) 
        self.linear2v = nn.Linear(50 + policy_size[0], policy_size[1]) 
        
        self.linear3p = nn.Linear(policy_size[1], policy_out_dim)
        self.linear3v = nn.Linear(policy_size[1], 1)
        
    def forward(self, ob):
        data_queue = ob[:,0:22].view(-1,1,2,11)
        ob_other = ob[:,22:].view(-1,12)
        
        action_logits = self.pool1p(F.relu(self.conv1p(data_queue)))
        action_logits = action_logits.view(-1,50)
        branch_b = F.relu(self.linear1p(ob_other))
        action_logits = torch.cat((action_logits, branch_b),dim=1)
        action_logits = F.relu(self.linear2p(action_logits))
        action_logits = self.linear3p(action_logits)
        
        dist = Categorical(logits = action_logits)
        
        value = self.pool1v(F.relu(self.conv1v(data_queue)))
        value = value.view(-1,50)
        value_b = F.relu(self.linear1v(ob_other))
        value = torch.cat((value, value_b),dim=1)
        value = F.relu(self.linear2v(value))
        value = self.linear3v(value)
        
        return dist, value
    

# In[]

class Agent(object):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args):
        self.ob_dim = computation_graph_args['ob_dim']
        self.ac_dim = computation_graph_args['ac_dim']
        self.extra_ob_dim = computation_graph_args['extra_ob_dim']
        self.num_users = computation_graph_args['num_users']
        self.num_agents = computation_graph_args['num_agents']
        self.discrete = computation_graph_args['discrete']
        self.CNN = computation_graph_args['CNN']
        self.size = computation_graph_args['size']
        self.size_critic = computation_graph_args['size_critic']
        self.n_layers = computation_graph_args['n_layers']
        self.learning_rate = computation_graph_args['learning_rate']
        self.recurrent = computation_graph_args['recurrent']
        self.input_prev_ac = computation_graph_args['input_prev_ac']
        self.ppo_epochs = computation_graph_args['ppo_epochs']
        self.mb_size = computation_graph_args['minibatch_size']
        self.ppo_clip = computation_graph_args['ppo_clip']
        self.v_coef = computation_graph_args['v_coef']
        self.entrop_loss_coef_schedule = computation_graph_args['entrop_loss_coef']
        self.max_grad_norm = computation_graph_args['max_grad_norm']
        self.test = computation_graph_args['test']

        self.animate = sample_trajectory_args['animate']
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']
        self.timesteps_this_batch = 0

        self.gamma = estimate_return_args['gamma']
        self.lamb = estimate_return_args['lambda']
        self.reward_to_go = estimate_return_args['reward_to_go']
        self.gae = estimate_return_args['gae']
        self.n_step = estimate_return_args['n_step']
        self.critic = estimate_return_args['critic']
        self.ex_time = estimate_return_args['ex_time']
        self.decentralised_critic = estimate_return_args['decentralised_critic']
        self.normalize_advantages = estimate_return_args['normalize_advantages']
        
        self.current_progress_remaining = 1.0
        
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device(0 if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        
        policy_in_dim = int(self.ob_dim)
        policy_out_dim = int(self.ac_dim)
        
        self.users_per_agent = int(self.num_users/self.num_agents)      # for parameter sharing
        
        self.logstd = {}
        self.net, self.optimizer = {}, {}
        self.log_prob_old = {}
        for i in range(self.num_agents):
            if self.CNN:
                self.net[str(i+1)] = CNN(policy_in_dim, policy_out_dim, self.size, self.size_critic, discrete=self.discrete).to(self.device)
            else:
                self.net[str(i+1)] = Network(policy_in_dim, policy_out_dim, self.size, self.size_critic, discrete=self.discrete).to(self.device)
            self.optimizer[str(i+1)] = optim.Adam(self.net[str(i+1)].parameters(), lr=self.learning_rate)
        
        
    def update_current_progress_remaining(self, num_iterations: int, total_iterations: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self.current_progress_remaining = 1.0 - float(num_iterations) / float(total_iterations)
        
    def update_entropy_loss(self):
        self.entrop_loss_coef = self.entrop_loss_coef_schedule(self.current_progress_remaining)
        
    """    
    Prefixes and suffixes:
            ob - observation 
            ac - action
            _no - this tensor should have shape (batch self.size /n/, observation dim)
            _na - this tensor should have shape (batch self.size /n/, action dim)
            _n  - this tensor should have shape (batch self.size /n/)
            _uo - (num users, ob dim)
            _nuo- (batch size, num users, ob dim)
            
    """
    def act(self, ob, h_=None, c_0=None):
        ob_uo, ac_ua, log_prob_ua, ac_np = {}, {}, {}, {}
        
        for i in range(self.num_users):
            ob_uo[str(i+1)] = torch.tensor(ob[str(i+1)],requires_grad=False,dtype=torch.float32, device=self.device) #.unsqueeze(0) 
            
            j = int(np.ceil((i+1)/self.users_per_agent))
            dist, value = self.net[str(j)](ob_uo[str(i+1)])         # torch.size([5]), ([1,5])
            
            ac_ua[str(i+1)] = dist.sample()                             # torch.size([1,5])
            log_prob_ua[str(i+1)] = dist.log_prob(ac_ua[str(i+1)])      # sum/avg only after clamping
            
            ac_np[str(i+1)] = ac_ua[str(i+1)].item()
        
        return ac_np, log_prob_ua
    
    
    def eval_ac(self, ob, ac_ua, idx, h_=None):
        ob_uo, log_prob_ua, entropy, ac_ua2, values = {}, {}, {}, {}, {}
        
        for i in range(self.num_users):
            ob_uo[str(i+1)] = torch.tensor(ob[str(i+1)][idx],requires_grad=False,dtype=torch.float32, device=self.device)
            ac_ua2[str(i+1)] = torch.tensor(ac_ua[str(i+1)][idx],requires_grad=False,dtype=torch.float32, device=self.device)
            
            j = int(np.ceil((i+1)/self.users_per_agent))
            dist, values[str(i+1)] = self.net[str(j)](ob_uo[str(i+1)])  # torch.Size([batch,5]), torch.Size([batch,1])
            
            log_prob_ua[str(i+1)] = dist.log_prob(ac_ua2[str(i+1)])     # torch.Size([batch])
            entropy[str(i+1)] = dist.entropy().mean()                   # torch.Size([])
            
        return values, log_prob_ua, entropy
    
    
    def eval_target(self, next_ob, re_n, terminal_n):
        next_ob_no, re_n2, next_v_n, target_n = {}, {}, {}, {}
        
        terminal_n2 = torch.tensor(terminal_n,requires_grad=False,dtype=torch.float32, device=self.device).reshape((-1,1))
        
        for i in range(self.num_users):
            next_ob_no[str(i+1)] = torch.tensor(next_ob[str(i+1)],requires_grad=False,dtype=torch.float32, device=self.device)
            re_n2[str(i+1)] = torch.tensor(re_n[str(i+1)],requires_grad=False,dtype=torch.float32, device=self.device).reshape((-1,1))
            
            j = int(np.ceil((i+1)/self.users_per_agent))
            with torch.no_grad():
                _, next_v_n[str(i+1)] = self.net[str(j)](next_ob_no[str(i+1)])
            target_n[str(i+1)] = (re_n2[str(i+1)] + (1 - terminal_n2) * self.gamma * next_v_n[str(i+1)]).detach()
        return target_n
    
    
    def sample_trajectories(self, itr, env):
        # Collect paths until we have enough timesteps
        self.timesteps_this_batch = 0
        paths = []
        while True:
            animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and self.animate)
            path = self.sample_trajectory(env, animate_this_episode)
            paths.append(path)
            self.timesteps_this_batch += pathlength(path['1'])
            if self.timesteps_this_batch >= self.min_timesteps_per_batch:
                break
        return paths, self.timesteps_this_batch
    
    
    def sample_trajectory(self, env, animate_this_episode):
        ob = env.reset()                    # returns ob['agent_no'] = 
        obs, acs, log_probs, rewards, next_obs, next_acs, hiddens, h_, entropys = {}, {}, {}, {}, {}, {}, {}, {}, {}
        terminals = []
        for i in range(self.num_users):
            obs[str(i+1)], acs[str(i+1)], log_probs[str(i+1)], rewards[str(i+1)], next_obs[str(i+1)], next_acs[str(i+1)], hiddens[str(i+1)], entropys[str(i+1)] = \
            [], [], [], [], [], [], [], []
        
        steps = 0
        
        ac, log_prob = self.act(ob) # YOUR CODE HERE
        
        while True:
            if animate_this_episode:
                env.render()
                time.sleep(0.1)            
            
            for i in range(self.num_users):
                obs[str(i+1)].append(ob[str(i+1)])
                acs[str(i+1)].append(ac[str(i+1)])
                log_probs[str(i+1)].append(log_prob[str(i+1)].detach())
            
            ob, rew, done, _ = env.step(ac)
            
            ac, log_prob = self.act(ob)
            
            for i in range(self.num_users):
                next_obs[str(i+1)].append(ob[str(i+1)])
                next_acs[str(i+1)].append(ac[str(i+1)])

            for i in range(self.num_users):
                rewards[str(i+1)].append(rew[str(i+1)])     # most recent reward appended to END of list
            
            steps += 1
            if done or steps >= self.max_path_length:
                terminals.append(1)
                break
            else:
                terminals.append(0)
        
        path = {}
        for i in range(self.num_users):
            path[str(i+1)] = {"observation" : np.array(obs[str(i+1)], dtype=np.float32).reshape(steps,-1),    # (steps,20)
                 "reward" : np.array(rewards[str(i+1)], dtype=np.float32),                  # (steps)
                 "log_prob": torch.cat(log_probs[str(i+1)]),                              # torch.size([steps, ac_dim])
                 "next_observation": np.array(next_obs[str(i+1)], dtype=np.float32).reshape(steps,-1),
                 "next_action": np.array(next_acs[str(i+1)], dtype=np.float32),
                 }
            if self.recurrent:
                path[str(i+1)]["hidden"] = torch.cat(hiddens[str(i+1)],dim=1)               # torch.size([1,steps,32])
                path[str(i+1)]["action"] = np.concatenate(acs[str(i+1)], axis = -2)
            else:
                path[str(i+1)]["action"] = np.array(acs[str(i+1)], dtype=np.float32)

        path["terminal"] = np.array(terminals, dtype=np.float32)
        
        # Log additional statistics for user #1
        path['nb_unexpected_ev'] = (env.episode_observation['unexpected_ev_counter'])
        path['wrong_mode_actions'] = (env.episode_observation['wrong_mode_actions'])
        path['throughput'] = (env.episode_observation['throughput'] / 400)
        
        path['data_counter'] = (env.episode_observation['data_counter'])
        path['urgency_counter'] = (env.episode_observation['urgency_counter'])
        path['peak_age_counter'] = (env.episode_observation['peak_age_counter'])
        path['comm_counter'] = (env.episode_observation['comm_counter'])
        path['radar_counter'] = (env.episode_observation['radar_counter'])
        path['comm_req_counter'] = (env.episode_observation['comm_req_counter'])
        path['radar_req_counter'] = (env.episode_observation['radar_req_counter'])
        path['good_ch_comm'] = (env.episode_observation['good_ch_comm'])
        path['r_age'] = (env.episode_observation['r_class_age'])
        path['r_radar'] = (env.episode_observation['r_radar'])
        path['r_overflow'] = (env.episode_observation['r_overflow'])
        
        if self.test:
            path['comm_req_t_idle_map'] = (env.episode_observation['comm_req_t_idle_map'])
            path['radar_req_t_idle_map'] = (env.episode_observation['radar_req_t_idle_map'])
            path['comm_req_t_last_map'] = (env.episode_observation['comm_req_t_last_map'])
            path['radar_req_t_last_map'] = (env.episode_observation['radar_req_t_last_map'])
            
            path['state_map'] = (env.episode_observation['state_map'])
            path['action_map'] = (env.episode_observation['action_map'])
            
            path['comm_req_channel_map'] = (env.episode_observation['comm_req_channel_map'])
            path['radar_req_channel_map'] = (env.episode_observation['radar_req_channel_map'])
            
            path['comm_req_channel_age_map'] = (env.episode_observation['comm_req_channel_age_map'])
            path['radar_req_channel_age_map'] = (env.episode_observation['radar_req_channel_age_map'])
            
            path['t_idle_map'] = (env.episode_observation['t_idle_map'])
            path['t_last_map'] = (env.episode_observation['t_last_map'])
            path['channel_age_map'] = (env.episode_observation['channel_age_map'])
        
        return path
    
    def estimate_decen_adv(self, ob_no, next_ob_no, re_n, terminal_n):       
        adv_n, ob_no2, next_ob_no2, re_n2 = {}, {}, {}, {}
        lastgaelam = 0
        
        for k in range(self.num_users):
            ob_no2[str(k+1)] = torch.tensor(ob_no[str(k+1)], requires_grad=False,dtype=torch.float32, device=self.device)
            next_ob_no2[str(k+1)] = torch.tensor(next_ob_no[str(k+1)], requires_grad=False,dtype=torch.float32, device=self.device)
            re_n2[str(k+1)] = torch.tensor(re_n[str(k+1)], requires_grad=False,dtype=torch.float32, device=self.device)
            adv_n[str(k+1)] = torch.zeros_like(re_n2[str(k+1)]).to(self.device)
            for t in reversed(range(self.min_timesteps_per_batch)):
                if t == self.min_timesteps_per_batch-1:
                    nextnonterminal = 1 - 1
                else:
                    nextnonterminal = 1 - terminal_n[t+1]
                j = int(np.ceil((k+1)/self.users_per_agent))
                with torch.no_grad():
                    _, v_next = self.net[str(j)](next_ob_no2[str(k+1)][t,:].reshape(-1,self.ob_dim))
                    _, v = self.net[str(j)](ob_no2[str(k+1)][t,:].reshape(-1,self.ob_dim))
                q = re_n2[str(k+1)][t] + self.gamma * v_next * nextnonterminal
                delta = q - v
                adv_n[str(k+1)][t] = lastgaelam = delta + self.gamma*self.lamb*nextnonterminal*lastgaelam
            if self.normalize_advantages:
                adv_n[str(k+1)] = (adv_n [str(k+1)]- torch.mean(adv_n[str(k+1)])) / (torch.std(adv_n[str(k+1)]) + 1e-7)
        return adv_n
    
    
    def update(self,ob_no, ac_na, adv_n, log_prob_na_old, next_ob_no, re_n, terminal_n, update=0, h_ns1=None):      
        policy_loss, policy_loss_after = {}, {}
        num_mb = int(np.ceil(self.timesteps_this_batch / self.mb_size))     # num minibatches
        
        loss, v_loss, v_loss_after = {}, {}, {}
        loss_agent = {}
        losses, pg_losses, clip_fractions, kl_divs, approx_kl_divs, value_losses, entropy_losses = {}, {}, {}, {}, {}, {}, {}
        losses_agent = {}
        v_loss_av = 0
        
        
        for i in range(1, self.num_users+1):
            losses[str(i)], pg_losses[str(i)], clip_fractions[str(i)], kl_divs[str(i)], value_losses[str(i)], entropy_losses[str(i)] = [], [], [], [], [], []
        for j in range(1, self.num_agents+1):
            loss_agent[str(j)] = 0
            losses_agent[str(j)] = 0
        
        for epoch in range(self.ppo_epochs):
            shuffle_idx = np.random.permutation(self.timesteps_this_batch)
            mb_idx = np.arange(num_mb)
            np.random.shuffle(mb_idx)
            
            target_n = self.eval_target(next_ob_no, re_n, terminal_n)
            
            for i in range(1, self.num_users+1):
                approx_kl_divs[str(i)] = []
            
            for k in mb_idx:
                idx = shuffle_idx[k*self.mb_size : (k+1)*self.mb_size]
                values, log_prob_na, entropy = self.eval_ac(ob_no, ac_na, idx, h_ns1)
                

                for i in range(self.num_users):
                    policy_loss[str(i+1)] = 0
                    policy_loss_after[str(i+1)] = 0
                    
                    # Calc policy loss
                    ratio = (log_prob_na[str(i+1)] - log_prob_na_old[str(i+1)][idx].detach()).exp()        # torch.Size([mb_size])
                    obj = torch.mul(ratio, adv_n[str(i+1)][idx])                                                # torch.Size([mb_size])
                    obj_clipped = torch.mul(torch.clamp(ratio, 1-self.ppo_clip, 1+self.ppo_clip), adv_n[str(i+1)][idx])     # torch.size([mb_size])
                    policy_loss[str(i+1)] = - torch.min(obj, obj_clipped).mean()                                # mean(torch.size([mb_size]))
                    
                    # Calc value loss
                    v_loss[str(i+1)] = (values[str(i+1)] - target_n[str(i+1)][idx]).pow(2).mean()        #
                
                    loss[str(i+1)] = policy_loss[str(i+1)] + self.v_coef * v_loss[str(i+1)] - self.entrop_loss_coef * entropy[str(i+1)]
                    j = int(np.ceil((i+1)/self.users_per_agent))
                    loss_agent[str(j)] = loss_agent[str(j)] + loss[str(i+1)]
                    
                    # Logging
                    losses[str(i+1)].append(loss[str(i+1)].item())
                    pg_losses[str(i+1)].append(policy_loss[str(i+1)].item())
                    clip_fraction = torch.mean((torch.abs(ratio - 1) > self.ppo_clip).float()).item()
                    clip_fractions[str(i+1)].append(clip_fraction)
                    approx_kl_divs[str(i+1)].append(torch.mean(log_prob_na_old[str(i+1)][idx]- log_prob_na[str(i+1)]).item())
                    value_losses[str(i+1)].append(v_loss[str(i+1)].item())
                    entropy_losses[str(i+1)].append(entropy[str(i+1)].item())
                
                for j in range(self.num_agents):
                    self.optimizer[str(j+1)].zero_grad()
                    loss_agent[str(j+1)].backward()
                    nn.utils.clip_grad_norm_(self.net[str(j+1)].parameters(), self.max_grad_norm)
                    self.optimizer[str(j+1)].step()
                    loss_agent[str(j+1)] = 0
                    
        # Log
        for i in range(1, self.num_users+1):
            logger.log_tabular("Loss "+str(i), np.mean(losses[str(i)]))
            logger.log_tabular("Policy Gradient Loss "+str(i), np.mean(pg_losses[str(i)]))
            logger.log_tabular("Clip Fraction "+str(i), np.mean(clip_fractions[str(i)]))
            logger.log_tabular("KL Divergence "+str(i), np.mean(approx_kl_divs[str(i)]))
            logger.log_tabular("Value Loss "+str(i), np.mean(value_losses[str(i)]))
            logger.log_tabular("Entropy Loss "+str(i), np.mean(entropy_losses[str(i)]))
            

# In[]
        
def train_AC(
        exp_name,
        env_name,
        env_config,
        ex_time,
        n_iter, 
        gamma,
        lamb,
        min_timesteps_per_batch, 
        max_path_length,
        learning_rate, 
        reward_to_go, 
        gae,
        n_step,
        animate, 
        logdir, 
        normalize_advantages,
        critic,
        decentralised_critic,
        seed,
        n_layers,
        conv,
        size,
        size_critic,
        recurrent,
        input_prev_ac,
        ppo_epochs,
        minibatch_size,
        ppo_clip,
        v_coef,
        entrop_loss_coef,
        entrop_const,
        max_grad_norm,
        policy_net_dir=None,
        value_net_dir=None,
        test = None):
    
    start = time.time()
    setup_logger(logdir, locals())  # setup logger for results
    
    env = AV_Environment(env_config)
    
    env.seed(seed)
    torch.manual_seed(seed)
#    np.random.seed(seed)
    
    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps
    
    # Is this env continuous, or self.discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    
    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]                                 # OK
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]  # OK
    extra_ob_dim = env.extra_ob_dim
    
    num_users = int(env.N)
    
    if entrop_const:
        entrop_loss_coef = constant_schedule(entrop_loss_coef)
    else:
        entrop_loss_coef = linear_schedule(entrop_loss_coef)
    
    computation_graph_args = {
        'n_layers': n_layers,
        'ob_dim': ob_dim,
        'ac_dim': ac_dim,
        'extra_ob_dim': extra_ob_dim,
        'num_users': env_config['num_users'],
        'num_agents': env_config['num_agents'],
        'discrete': discrete,
        'size': size,
        'CNN': conv,
        'size_critic': size_critic,
        'learning_rate': learning_rate,
        'recurrent': recurrent,
        'input_prev_ac': input_prev_ac,
        'ppo_epochs': ppo_epochs,
        'minibatch_size': minibatch_size,
        'ppo_clip': ppo_clip,
        'v_coef': v_coef,
        'entrop_loss_coef': entrop_loss_coef,
        'max_grad_norm': max_grad_norm,
        'test': test,
        }

    sample_trajectory_args = {
        'animate': animate,
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }

    estimate_return_args = {
        'gamma': gamma,
        'lambda': lamb,
        'reward_to_go': reward_to_go,
        'gae': gae,
        'n_step': n_step,
        'critic': critic,
        'ex_time': ex_time,
        'decentralised_critic': decentralised_critic,
        'normalize_advantages': normalize_advantages,
    }
    
    agent = Agent(computation_graph_args, sample_trajectory_args, estimate_return_args)
    
    # policy_net_dir = os.path.join('data',
    #                               'Multi_v0_PPO2.6_4usr1agnt_entrop_loss_decay0.01_lr1e-4_JRCAoI_multi_v0d2_29-03-2021_17-58-55_rdbad_1.1_wr5_gen1.0_nn[64, 64]',
    #                               '1',
    #                               'NN1_itr788')
    test = False
    if policy_net_dir != None:
        for i in range(agent.num_agents):
            load_variables(agent.net[str(i+1)], policy_net_dir)
        test = True
    
        
    #========================================================================================#
    # Training Loop
    #========================================================================================#
    
    total_timesteps = 0
    best_av_reward = None
    policy_model_file = {}
    for itr in range(n_iter):
        print("********** Iteration %i ************"%itr)
        paths, timesteps_this_batch = agent.sample_trajectories(itr, env)
        total_timesteps += timesteps_this_batch
        
        agent.update_current_progress_remaining(itr, n_iter)
        agent.update_entropy_loss()

        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no, ac_na, re_n, log_prob_na, next_ob_no, next_ac_na, h_ns1, entropy = {}, {}, {}, {}, {}, {}, {}, {}
        returns = np.zeros((num_users,len(paths)))
        for i in range(num_users):
            ob_no[str(i+1)] = np.concatenate([path[str(i+1)]["observation"] for path in paths])         # shape (batch self.size /n/, ob_dim)
            ac_na[str(i+1)] = np.concatenate([path[str(i+1)]["action"] for path in paths]) #, axis = -2)   # shape (batch self.size /n/, ac_dim) recurrent: (1, n, ac_dim)
            re_n[str(i+1)] = np.concatenate([path[str(i+1)]["reward"] for path in paths])               # (batch_size, num_users)
            log_prob_na[str(i+1)] = torch.cat([path[str(i+1)]["log_prob"] for path in paths])           # torch.size([5200, ac_dim])
            next_ob_no[str(i+1)] = np.concatenate([path[str(i+1)]["next_observation"] for path in paths]) # shape (batch self.size /n/, ob_dim)
            next_ac_na[str(i+1)] = np.concatenate([path[str(i+1)]["next_action"] for path in paths]) # shape (batch self.size /n/, ac_dim)
            if agent.recurrent:
                h_ns1[str(i+1)] = torch.cat([path[str(i+1)]["hidden"] for path in paths],dim=1)    #torch.size([1, batchsize, 32])
            returns[i,:] = [path[str(i+1)]["reward"].sum(dtype=np.float32) for path in paths]   # (num_users, num episodes in batch)
        
        terminal_n = np.concatenate([path["terminal"] for path in paths])       # (batch_size,)
        
        nb_unexpected_ev = ([path["nb_unexpected_ev"] for path in paths])
        wrong_mode_actions = ([path["wrong_mode_actions"] for path in paths])              # (batch,)
        throughput = ([path["throughput"] for path in paths])
        
        data_counter = ([path["data_counter"] for path in paths])
        urgency_counter = ([path["urgency_counter"] for path in paths])
        peak_age_counter = ([path["peak_age_counter"] for path in paths])
        comm_counter = ([path["comm_counter"] for path in paths])
        radar_counter = ([path["radar_counter"] for path in paths])
        comm_req_counter = np.array([path["comm_req_counter"] for path in paths])
        radar_req_counter = np.array([path["radar_req_counter"] for path in paths])
        if test==True:
            comm_req_t_idle_map = np.array([path["comm_req_t_idle_map"] for path in paths])
            radar_req_t_idle_map = np.array([path["radar_req_t_idle_map"] for path in paths])
            comm_req_t_last_map = np.array([path["comm_req_t_last_map"] for path in paths])
            radar_req_t_last_map = np.array([path["radar_req_t_last_map"] for path in paths])
            state_map = np.array([path["state_map"] for path in paths])
            action_map = np.array([path["action_map"] for path in paths])
            comm_req_channel_map = np.array([path["comm_req_channel_map"] for path in paths])
            radar_req_channel_map = np.array([path["radar_req_channel_map"] for path in paths])
            comm_req_channel_age_map = np.array([path["comm_req_channel_age_map"] for path in paths])
            radar_req_channel_age_map = np.array([path["radar_req_channel_age_map"] for path in paths])
            
            t_idle_map = np.array([path["t_idle_map"] for path in paths])
            t_last_map = np.array([path["t_last_map"] for path in paths])
            channel_age_map = np.array([path["channel_age_map"] for path in paths])
        good_ch_comm = ([path["good_ch_comm"] for path in paths])
        r_age = ([path["r_age"] for path in paths])
        r_radar = ([path["r_radar"] for path in paths])
        r_overflow = ([path["r_overflow"] for path in paths])
        
        av_reward = np.mean(returns)
        
        # Log diagnostics
        ep_lengths = [pathlength(path['1']) for path in paths]
        logger.log_tabular("Time", time.time() - start)
        logger.log_tabular("Iteration", itr)
        logger.log_tabular("Average Reward", av_reward)   # per agent per episode
        logger.log_tabular("StdReward", np.std(returns))
        logger.log_tabular("MaxReward", np.max(returns))
        logger.log_tabular("MinReward", np.min(returns))
        logger.log_tabular("nb_unexpected_ev", np.mean(nb_unexpected_ev))
        logger.log_tabular("wrong_mode_actions", np.mean(wrong_mode_actions))
        logger.log_tabular("comm action %", np.mean(comm_counter)/400)
        logger.log_tabular("radar action %", np.mean(radar_counter)/400)
        logger.log_tabular("no-op %", (400 - np.mean(comm_counter) - np.mean(radar_counter)) / 400)
        if test==True:
            logger.log_tabular("comm action req %", np.mean(comm_req_counter, axis=0)/400)
            logger.log_tabular("radar action req %", np.mean(radar_req_counter, axis=0)/400)
            logger.log_tabular("comm_req_t_idle_map", np.mean(comm_req_t_idle_map, axis=0) / 400)
            logger.log_tabular("radar_req_t_idle_map", np.mean(radar_req_t_idle_map, axis=0) / 400)
            logger.log_tabular("comm_req_t_last_map", np.mean(comm_req_t_last_map, axis=0) / 400)
            logger.log_tabular("radar_req_t_last_map", np.mean(radar_req_t_last_map, axis=0) / 400)
            logger.log_tabular("state_map", np.mean(state_map, axis=0) / 400)
            logger.log_tabular("action_map", np.mean(action_map, axis=0) / 400)
            logger.log_tabular('comm_req_channel_map', np.mean(comm_req_channel_map, axis=0) / 400)
            logger.log_tabular('radar_req_channel_map', np.mean(radar_req_channel_map, axis=0) / 400)
            logger.log_tabular('comm_req_channel_age_map', np.mean(comm_req_channel_age_map, axis=0) / 400)
            logger.log_tabular('radar_req_channel_age_map', np.mean(radar_req_channel_age_map, axis=0) / 400)
            
            logger.log_tabular('t_idle_map', np.mean(t_idle_map, axis=0) / 400)
            logger.log_tabular('t_last_map', np.mean(t_last_map, axis=0) / 400)
            logger.log_tabular('channel_age_map', np.mean(channel_age_map, axis=0) / 400)
        else:
            logger.log_tabular("comm action req %", np.mean(comm_req_counter)/400)
            logger.log_tabular("radar action req %", np.mean(radar_req_counter)/400)
        logger.log_tabular("no-op  req %", (400 - np.mean(comm_req_counter) - np.mean(radar_req_counter)) / 400)
        logger.log_tabular("throughput", np.mean(throughput))
        logger.log_tabular("r_age", np.mean(r_age))
        logger.log_tabular("r_radar", np.mean(r_radar))
        logger.log_tabular("r_overflow", np.mean(r_overflow))
        for i in range(num_users):
            logger.log_tabular("Reward"+str(i+1), np.mean(returns, axis=1)[i])
            logger.log_tabular("StdReward"+str(i+1), np.std(returns, axis=1)[i])
            logger.log_tabular("MaxReward"+str(i+1), np.max(returns, axis=1)[i])
            logger.log_tabular("MinReward"+str(i+1), np.min(returns, axis=1)[i])
        logger.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logger.log_tabular("TimestepsSoFar", total_timesteps)
        
        if test != True:
            adv_n = agent.estimate_decen_adv(ob_no, next_ob_no, re_n, terminal_n)
            agent.update(ob_no, ac_na, adv_n, log_prob_na, next_ob_no, re_n, terminal_n)
        
            if best_av_reward == None:
                best_av_reward = av_reward
            elif av_reward > best_av_reward:
                best_av_reward = av_reward
                for i in range(env_config['num_agents']):
                    policy_model_file[str(i+1)] = os.path.join(logdir,"NN"+str(i+1)+'_itr'+str(itr))
                    save_itr_info(f"{policy_model_file[str(i+1)]}-{itr}.txt", itr, av_reward)
                    save_variables(agent.net[str(i+1)], policy_model_file[str(i+1)])
        
            logger.dump_tabular(step=itr)
        else:
            logger.save_data()
    
    
    if test != True:
        for i in range(env_config['num_agents']):
            policy_model_file[str(i+1)] = os.path.join(logdir,"NN"+str(i+1))
            save_itr_info(f"{policy_model_file[str(i+1)]}-{itr}.txt", itr, av_reward)
            save_variables(agent.net[str(i+1)], policy_model_file[str(i+1)])
        
    

# In[]

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    # Env config
    parser.add_argument('env_name', type=str)
    parser.add_argument('--num_users', type=int)
    parser.add_argument('--num_agents', type=int)
    parser.add_argument('--obj', choices=['peak','avg'], default='peak')
    parser.add_argument('--w_radar', type=int, nargs='+', default=[0,10,5])
    parser.add_argument('--w_ovf', type=float, default=0)
    parser.add_argument('--w_age', type=float, default=None)
    parser.add_argument('--phi', type=float, default=1000)
    parser.add_argument('--pv', type=int, nargs='+', default=[1,2,1])
    parser.add_argument('--data_gen', type=float, nargs='+', default=[2,3,1])
    parser.add_argument('--rd_bad2bad', type=float, nargs='+', default=[0.1,0.2,0.1])
    parser.add_argument('--ex_price', action='store_true')
    parser.add_argument('--ex_time', action='store_true')
    
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    
    # Algorithm hyperparameters
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--lamb', type=float, default=0.95)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=3e-4)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--generalised_adv_est', '-gae', action='store_true')
    parser.add_argument('--n_step', '-ns', type=int, default=0)
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--critic', type=str, default ='v')
    parser.add_argument('--decentralised_critic', '-dc', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    parser.add_argument('--conv', action='store_true')
    parser.add_argument('--size', '-s', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--size_critic', '-sc', type=int, nargs='+', default=[64, 64])
    parser.add_argument('--recurrent', '-r', action='store_true')
    parser.add_argument('--input_prev_ac','-ipa', action='store_true')
    parser.add_argument('--ppo_epochs', type=int, default=10)
    parser.add_argument('--minibatch_size', '-mbs', type=int, default=64)
    parser.add_argument('--ppo_clip', type=float, default=0.2) #0.05
    parser.add_argument('--v_coef', type=float, default=0.5)
    parser.add_argument('--entrop_loss_coef', type=float, default=0) #0.001) #0.0005
    parser.add_argument('--entrop_const', action='store_true')
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--ac_dist', default='Gaussian')
    
    
    parser.add_argument('--policy_net_filename', '-p_file', type=str)
    parser.add_argument('--value_net_filename', '-v_file', type=str)
    parser.add_argument('--test', '-t', action='store_true')
    parser.add_argument('--pre_uuid', '-uuid', type=str)
    args = parser.parse_args()
    
    
    if not(os.path.exists('data')):
        os.makedirs('data')
    
    if args.pre_uuid != None:
        logdir = args.exp_name + '_' + args.env_name + '_' + args.pre_uuid
        logdir = os.path.join('data/' + args.pre_uuid , logdir)
    else:
        logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
        logdir = os.path.join('data', logdir)
    
    print("------")
    print(logdir)
    print("------")
    
       
    max_path_length = args.ep_len if args.ep_len > 0 else None
    args.policy_net_dir = None
    args.value_net_dir = None
    
    # if args.policy_net_filename != None:
    #     args.policy_net_dir = os.path.join(os.getcwd(),args.policy_net_filename)
    # if args.value_net_filename != None:
    #     args.value_net_dir = os.path.join(os.getcwd(),args.value_net_filename)
    
    
    for data_gen in np.arange(args.data_gen[0], args.data_gen[1], args.data_gen[2]):
        for w_radar in range(args.w_radar[0], args.w_radar[1], args.w_radar[2]):
            for pv in range(args.pv[0], args.pv[1], args.pv[2]):
                for rd_bad2bad in np.arange(args.rd_bad2bad[0],args.rd_bad2bad[1],args.rd_bad2bad[2]):
                    for e in range(args.n_experiments):
                        
                        if data_gen == 1.1:
                            break
                        
                        seed = args.seed + 10*e
                        print('Running experiment with seed %d'%seed)
                        
                        test = False
                        if args.policy_net_filename != None:
                            test = True
                        
                        env_config = {'num_users': args.num_users,
                                      'num_agents': args.num_agents,
                                      'pv': pv/10,
                                      'w_age': args.w_age,
                                      'w_radar': w_radar,
                                      'w_overflow': args.w_ovf,
                                      'data_gen': float(data_gen),
                                      'road_sw_bad_to_bad': float(rd_bad2bad),
                                      'age_obj': args.obj,
                                      'phi': args.phi,
                                      'test': test}
                        
                        logdir_w_params = logdir + "_rdbad_{}_wr{}_gen{}_nn{}".format(rd_bad2bad, w_radar, data_gen, args.size)
                        
                        train_AC(
                                exp_name=args.exp_name,
                                env_name=args.env_name,
                                env_config=env_config,
                                ex_time = args.ex_time,
                                n_iter=args.n_iter,
                                gamma=args.discount,
                                lamb=args.lamb,
                                min_timesteps_per_batch=args.batch_size,
                                max_path_length=max_path_length,
                                learning_rate=args.learning_rate,
                                reward_to_go=args.reward_to_go,
                                gae = args.generalised_adv_est,
                                n_step = args.n_step,
                                animate=args.render,
                                logdir=os.path.join(logdir_w_params,'%d'%seed),
                                normalize_advantages=not(args.dont_normalize_advantages),
                                critic=args.critic,
                                decentralised_critic= args.decentralised_critic,
                                seed=seed,
                                n_layers=args.n_layers,
                                size=args.size,
                                conv=args.conv,
                                size_critic=args.size_critic,
                                recurrent = args.recurrent,
                                input_prev_ac = args.input_prev_ac,
                                ppo_epochs = args.ppo_epochs,
                                minibatch_size = args.minibatch_size,
                                ppo_clip = args.ppo_clip,
                                v_coef = args.v_coef,
                                entrop_loss_coef = args.entrop_loss_coef,
                                entrop_const = args.entrop_const,
                                max_grad_norm = args.max_grad_norm,
                                policy_net_dir = args.policy_net_filename,
                                value_net_dir = args.value_net_filename,
                                test = args.test
                                )