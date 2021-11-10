#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%pdb
"""
Code for the paper "Learning to Schedule Joint Radar-Communication with Deep
Multi-Agent Reinforcement Learning", IEEE Transactions on Vehicular Technology

Author: Joash Lee

This program uses the Multi-Agent Advantage Actor-Critic (A2C) algorithm
to solve the Joint Radar-Communication (JRC) and Age of Information (AoI) Markov
Game in the file 'JRCwithAOI_multi.py'


"""

import numpy as np
import torch
import torch.nn as nn
import gym
import logger
import os
import time
import inspect
from typing import Callable, Union

from trainPPO_JRC import Agent

from JRCwithAOI_multi import AV_Environment


# In[]

device = torch.device(0 if torch.cuda.is_available() else "cpu")
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



# In[]

class A2CAgent(Agent):
    def __init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args):
        Agent.__init__(self, computation_graph_args, sample_trajectory_args, estimate_return_args)
        
        
    def update(self,ob_no, ac_na, adv_n, log_prob_na_old, next_ob_no, re_n, terminal_n, update=0, h_ns1=None):      
        policy_loss, policy_loss_after = {}, {}
        num_mb = int(np.ceil(self.timesteps_this_batch / self.mb_size))     # num minibatches
        
        loss, v_loss, v_loss_after = {}, {}, {}
        loss_agent = {}
        losses, pg_losses, kl_divs, approx_kl_divs, value_losses, entropy_losses = {}, {}, {}, {}, {}, {}
        losses_agent = {}
        v_loss_av = 0
        
        
        for i in range(1, self.num_users+1):
            losses[str(i)], pg_losses[str(i)], kl_divs[str(i)], value_losses[str(i)], entropy_losses[str(i)] = [], [], [], [], []
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
                    policy_loss[str(i+1)] = - torch.mul(log_prob_na[str(i+1)], adv_n[str(i+1)][idx]).mean()
                    
                    # Calc value loss
                    v_loss[str(i+1)] = (values[str(i+1)] - target_n[str(i+1)][idx]).pow(2).mean()
                    v_loss_av = v_loss_av + v_loss[str(i+1)].detach()/self.num_users
                
                    loss[str(i+1)] = policy_loss[str(i+1)] + self.v_coef * v_loss[str(i+1)] - self.entrop_loss_coef * entropy[str(i+1)]
                    j = int(np.ceil((i+1)/self.users_per_agent))
                    loss_agent[str(j)] = loss_agent[str(j)] + loss[str(i+1)]
                    
                    # Logging
                    losses[str(i+1)].append(loss[str(i+1)].item())
                    pg_losses[str(i+1)].append(policy_loss[str(i+1)].item())
                    approx_kl_divs[str(i+1)].append(torch.mean(log_prob_na_old[str(i+1)][idx]- log_prob_na[str(i+1)]).item())
                    value_losses[str(i+1)].append(v_loss[str(i+1)].item())
                    entropy_losses[str(i+1)].append(entropy[str(i+1)].item())
                
                for j in range(self.num_agents):
                    self.optimizer[str(j+1)].zero_grad()
                    loss_agent[str(j+1)].backward() #retain_graph = True)
                    nn.utils.clip_grad_norm_(self.net[str(j+1)].parameters(), self.max_grad_norm)
                    self.optimizer[str(j+1)].step()
                    loss_agent[str(j+1)] = 0
                
                    
        # Log
        for i in range(1, self.num_users+1):
            logger.log_tabular("Loss "+str(i), np.mean(losses[str(i)]))
            logger.log_tabular("Policy Gradient Loss "+str(i), np.mean(pg_losses[str(i)]))
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
    
    agent = A2CAgent(computation_graph_args, sample_trajectory_args, estimate_return_args)
    
    solved = False
    if policy_net_dir != None:
        load_variables(agent.net, policy_net_dir)
        solved = True
    
        
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
            assert ob_no[str(i+1)].shape == (timesteps_this_batch,ob_dim)
            assert ac_na[str(i+1)].shape == (timesteps_this_batch,)
            assert re_n[str(i+1)].shape == (timesteps_this_batch,)
            assert log_prob_na[str(i+1)].shape == torch.Size([timesteps_this_batch])
            assert next_ob_no[str(i+1)].shape == (timesteps_this_batch,ob_dim)
            assert next_ac_na[str(i+1)].shape == (timesteps_this_batch,)
            assert returns[i,:].shape == (timesteps_this_batch/400,)
        
        terminal_n = np.concatenate([path["terminal"] for path in paths])       # (batch_size,)
        
        nb_unexpected_ev = ([path["nb_unexpected_ev"] for path in paths])
        wrong_mode_actions = ([path["wrong_mode_actions"] for path in paths])              # (batch,)
        throughput = ([path["throughput"] for path in paths])
        
        data_counter = ([path["data_counter"] for path in paths])
        urgency_counter = ([path["urgency_counter"] for path in paths])
        peak_age_counter = ([path["peak_age_counter"] for path in paths])
        comm_counter = ([path["comm_counter"] for path in paths])
        radar_counter = ([path["radar_counter"] for path in paths])
        comm_req_counter = ([path["comm_req_counter"] for path in paths])
        radar_req_counter = ([path["radar_req_counter"] for path in paths])
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
        
        adv_n = agent.estimate_decen_adv(ob_no, next_ob_no, re_n, terminal_n)
        agent.update(ob_no, ac_na, adv_n, log_prob_na, next_ob_no, re_n, terminal_n)
        
        logger.dump_tabular(step=itr)
        
        if best_av_reward == None:
            best_av_reward = av_reward
        elif av_reward > best_av_reward:
            best_av_reward = av_reward
            for i in range(env_config['num_agents']):
                policy_model_file[str(i+1)] = os.path.join(logdir,"NN"+str(i+1)+'_itr'+str(itr))
                save_itr_info(f"{policy_model_file[str(i+1)]}-{itr}.txt", itr, av_reward)
                save_variables(agent.net[str(i+1)], policy_model_file[str(i+1)])
    
    
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
    
    if args.policy_net_filename != None:
        args.policy_net_dir = os.path.join(os.getcwd(),args.policy_net_filename)
    if args.value_net_filename != None:
        args.value_net_dir = os.path.join(os.getcwd(),args.value_net_filename)
    
    
    for data_gen in np.arange(args.data_gen[0], args.data_gen[1], args.data_gen[2]):
        for w_radar in range(args.w_radar[0], args.w_radar[1], args.w_radar[2]):
            for pv in range(args.pv[0], args.pv[1], args.pv[2]):
                for rd_bad2bad in np.arange(args.rd_bad2bad[0],args.rd_bad2bad[1],args.rd_bad2bad[2]):
                    for e in range(args.n_experiments):
                        
                        if data_gen == 1.1:
                            break
                        
                        seed = args.seed + 10*e
                        print('Running experiment with seed %d'%seed)
                        
                        env_config = {'num_users': args.num_users,
                                      'num_agents': args.num_agents,
                                      'pv': pv/10,
                                      'w_age': args.w_age,
                                      'w_radar': w_radar,
                                      'w_overflow': args.w_ovf,
                                      'data_gen': float(data_gen),
                                      'road_sw_bad_to_bad': float(rd_bad2bad),
                                      'age_obj': args.obj,
                                      'phi': args.phi,}
                        
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
                                conv=args.conv,
                                size=args.size,
                                size_critic=args.size_critic,
                                recurrent = args.recurrent,
                                input_prev_ac = args.input_prev_ac,
                                ppo_epochs = args.ppo_epochs,
                                minibatch_size = args.minibatch_size,
                                ppo_clip = args.ppo_clip,
                                v_coef = args.v_coef,
                                entrop_loss_coef = args.entrop_loss_coef,
                                max_grad_norm = args.max_grad_norm,
                                policy_net_dir = args.policy_net_dir,
                                value_net_dir = args.value_net_dir,
                                test = args.test
                                )