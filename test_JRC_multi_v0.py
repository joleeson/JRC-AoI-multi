from __future__ import division
# from JRCwithAOI_v0b import AV_Environment
# from JRCwithAOI_v3b import AV_Environment
# from JRCwithAOI_multi_v0 import AV_Environment
from JRCwithAOI_multi_v0d2 import AV_Environment
from config_jrc_aoi_multi_v0a import state_space_size
import numpy as np
import numpy.random as random
import random as python_random
# import json
import time
import os
import argparse
import logz
import inspect

"""

"""


def pathlength(path):
    return len(path["reward"])

def setup_logger(logdir, locals_):
    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    # Log experimental parameters
    args = inspect.getargspec(test)[0]
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

def alternative_switch_action(t, num_actions):
    """
    Alternates between communication '0' and a choice of communications actions.
    Cycles between the communication actions

    Parameters
    ----------
    t : time step
    num_actions : Tnumber of communication actions available

    Returns
    -------
    action num

    """
    if t % 2 == 1:
        return 0
    else:
        r = t % (num_actions*2)
        return int(r/2 + 1)

def alt_switch_action5(t, comm_action):
    """
    Alternates between communication '0' and communicating packets with urgency level 'comm_action'.

    Parameters
    ----------
    t : time step
    num_actions : Tnumber of communication actions available

    Returns
    -------
    action num

    """
    if t % 2 == 1:
        return 0
    else:
        return comm_action

class Agent(object):
    def __init__(self, policy_config, sample_trajectory_args, env_args):
        
        self.num_users = env_args['num_users']
        
        self.mode = policy_config['mode']
        self.ac_dim = policy_config['ac_dim']
        self.CW_min = policy_config['CW'][0]
        self.CW_max = policy_config['CW'][1]
        self.CW = np.ones((self.num_users,)) * self.CW_min
        self.counter = np.zeros((self.num_users,))
        
        self.max_path_length = sample_trajectory_args['max_path_length']
        self.min_timesteps_per_batch = sample_trajectory_args['min_timesteps_per_batch']
        
        self.timesteps_this_batch = 0
        
        self.counter = np.random.randint(self.CW, size=self.num_users)
    
    
    def act(self, ob):
        # if counter is zero select actions
        actions = (self.counter==0) * np.random.randint(self.ac_dim)
        action_reqs = actions
        
        priorities = np.empty((0),dtype=int)
        for n in range(self.num_users):
            priorities = np.concatenate((priorities, ob[str(n+1)][0,state_space_size['data_size']*2].reshape(1)))
        
        if np.sum(actions>0) > 1:
            if np.sum(priorities * (actions>0)) == 1:
                actions = (priorities == 1) * action_reqs   # Choose agent that transmits based on priority
            else:
                idxs = np.argwhere(actions>0).reshape(-1)
                idx = np.random.randint(len(idxs))
                action = actions[idx]
                actions = np.zeros_like(actions)
                actions[idx] = action
        unsuccessful_ac_reqs = (actions==0) * (action_reqs!=0)         # agents that act but unsuccessful
        
        # Halve CW for successful transmission, double for unsuccessful transmission
        self.CW = np.clip(((actions>0) * self.CW / 2) + ((actions==0) * self.CW), 2, self.CW_max)
        self.CW = np.clip((unsuccessful_ac_reqs * self.CW * 2) + (unsuccessful_ac_reqs==0 * self.CW), 2, self.CW_max)
        
        # decrement counter
        self.counter = np.clip(self.counter - 1, a_min=0, a_max=self.CW_max)
        # reset counter for agents that attempted to take action
        self.counter = ((action_reqs>0) * np.random.randint(self.CW, size=self.num_users)) + ((action_reqs==0) * self.counter)
        
        ac = {}
        for n in range(self.num_users):
            ac[str(n+1)] = actions[n]
            
        return ac
    
    
    def sample_trajectories(self, env):
        # Collect paths until we have enough timesteps
        self.timesteps_this_batch = 0
        paths = []
        while True:
            path = self.sample_trajectory(env)
            paths.append(path)
            self.timesteps_this_batch += pathlength(path['1'])
            if self.timesteps_this_batch >= self.min_timesteps_per_batch:
                break
        return paths, self.timesteps_this_batch
    
    
    def sample_trajectory(self, env):
        ob = env.reset()                    # returns ob['agent_no'] = 
        obs, acs, log_probs, rewards, next_obs, next_acs, hiddens, entropys = {}, {}, {}, {}, {}, {}, {}, {}
        terminals = []
        for i in range(self.num_users):
            obs[str(i+1)], acs[str(i+1)], log_probs[str(i+1)], rewards[str(i+1)], next_obs[str(i+1)], next_acs[str(i+1)], hiddens[str(i+1)], entropys[str(i+1)] = \
            [], [], [], [], [], [], [], []
        
        steps = 0
        
        for i in range(self.num_users):
            if self.mode == 'unif_rand':
                acs[str(i+1)] = np.array(np.random.randint(env.action_space.n))
            elif self.mode == 'urg5':
                acs[str(i+1)] = alt_switch_action5(steps, 5)
        if self.mode == 'csma-ca':
            acs = self.act(ob)

        
        while True:
            ob, rew, done, _ = env.step(acs)
            
            for i in range(self.num_users):
                rewards[str(i+1)].append(rew[str(i+1)])     # most recent reward appended to END of list
                
                if self.mode == 'unif_rand':
                    acs[str(i+1)] = np.array(np.random.randint(env.action_space.n))
                elif self.mode == 'urg5':
                    acs[str(i+1)] = alt_switch_action5(steps, 5)
            if self.mode == 'csma-ca':
                acs = self.act(ob)

                
            
            steps += 1
            if done or steps >= self.max_path_length:
                terminals.append(1)
                break
            else:
                terminals.append(0)
        
        path = {}
        for i in range(self.num_users):
            path[str(i+1)] = {"reward" : np.array(rewards[str(i+1)], dtype=np.float32),                  # (steps)
                 }
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
        path['radar_counter'] = (env.episode_observation['radar_counter'])
        path['comm_req_counter'] = (env.episode_observation['comm_req_counter'])
        path['radar_req_counter'] = (env.episode_observation['radar_req_counter'])
        path['good_ch_comm'] = (env.episode_observation['good_ch_comm'])
        path['r_age'] = (env.episode_observation['r_class_age'])
        path['r_radar'] = (env.episode_observation['r_radar'])
        path['r_overflow'] = (env.episode_observation['r_overflow'])
        
        return path


def test(
        exp_name,
        env_name,
        env_config,
        n_iter, 
        min_timesteps_per_batch, 
        max_path_length,
        seed,
        mode,
        CW,
        logdir,
        ):
    
    start = time.time()
    setup_logger(logdir, locals())  # setup logger for results    
    
    env = AV_Environment(env_config)
    env.seed(seed)
    
    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps
    policy_config = {'mode': args.mode,
                     'CW': CW,
                     'ac_dim': env.action_space.n,
                         }
    env_args = {'num_users': env_config['num_users']}
    sample_trajectory_args = {
        'max_path_length': max_path_length,
        'min_timesteps_per_batch': min_timesteps_per_batch,
    }
    agent = Agent(policy_config, sample_trajectory_args, env_args)
    
    total_timesteps = 0
    
    for itr in range(n_iter):
        paths, timesteps_this_batch = agent.sample_trajectories(env)
        total_timesteps += timesteps_this_batch
    
        # Build arrays for observation, action for the policy gradient update by concatenating 
        # across paths
        ob_no, ac_na, re_n, log_prob_na, next_ob_no, next_ac_na, h_ns1, entropy = {}, {}, {}, {}, {}, {}, {}, {}
        returns = np.zeros((args.num_users,len(paths)))
        for i in range(args.num_users):
            re_n[str(i+1)] = np.concatenate([path[str(i+1)]["reward"] for path in paths])               # (batch_size, num_users)
            returns[i,:] = [path[str(i+1)]["reward"].sum(dtype=np.float32) for path in paths]   # (num_users, num episodes in batch)
            assert re_n[str(i+1)].shape == (timesteps_this_batch,)
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
        
        
        logz.log_tabular("Time", time.time() - start)
        logz.log_tabular("Iteration", itr)
        logz.log_tabular("Average Reward", np.mean(returns))   # per agent per episode
        logz.log_tabular("StdReward", np.std(returns))
        logz.log_tabular("MaxReward", np.max(returns))
        logz.log_tabular("MinReward", np.min(returns))
        logz.log_tabular("nb_unexpected_ev", np.mean(nb_unexpected_ev))
        logz.log_tabular("wrong_mode_actions", np.mean(wrong_mode_actions))
        logz.log_tabular("comm action %", np.mean(comm_counter)/400)
        logz.log_tabular("radar action %", np.mean(radar_counter)/400)
        logz.log_tabular("no-op %", (400 - np.mean(comm_counter) - np.mean(radar_counter)) / 400)
        logz.log_tabular("comm action req %", np.mean(comm_req_counter)/400)
        logz.log_tabular("radar action req %", np.mean(radar_req_counter)/400)
        logz.log_tabular("no-op  req %", (400 - np.mean(comm_req_counter) - np.mean(radar_req_counter)) / 400)
        logz.log_tabular("throughput", np.mean(throughput))
        logz.log_tabular("r_age", np.mean(r_age))
        logz.log_tabular("r_radar", np.mean(r_radar))
        logz.log_tabular("r_overflow", np.mean(r_overflow))
        for i in range(env_config['num_users']):
            logz.log_tabular("Reward"+str(i+1), np.mean(returns, axis=1)[i])
            logz.log_tabular("StdReward"+str(i+1), np.std(returns, axis=1)[i])
            logz.log_tabular("MaxReward"+str(i+1), np.max(returns, axis=1)[i])
            logz.log_tabular("MinReward"+str(i+1), np.min(returns, axis=1)[i])
        logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
        logz.log_tabular("TimestepsSoFar", total_timesteps)
        
        logz.dump_tabular(step=itr)

parser = argparse.ArgumentParser()
parser.add_argument('env_name', type=str, default='AV_JRC_AoI-v3d')
parser.add_argument('--num_users', type=int)
parser.add_argument('--num_agents', type=int)
parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
parser.add_argument('--obj', choices=['peak','avg'], default='avg')
parser.add_argument('--w_radar', type=int, nargs='+', default=[0,10,1])
parser.add_argument('--w_ovf', type=float, default=1)
parser.add_argument('--w_age', type=float, default=None)
parser.add_argument('--phi', type=float, default=1000)
parser.add_argument('--pv', type=int, nargs='+', default=[1,2,1])
parser.add_argument('--data_gen', type=int, nargs='+', default=[3,4,1])
parser.add_argument('--rd_bad2bad', type=float, nargs='+', default=[0.1,0.2,0.1])
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--n_experiments', '-e', type=int, default=1)
parser.add_argument('--mode', choices=['unif_rand','urg5','best','csma-ca'], default='rotate')
parser.add_argument('--CW', type=int, nargs='+', default=[2,16])

parser.add_argument('--exp_name', type=str, default='vpg')

parser.add_argument('--n_iter', '-n', type=int, default=100)
parser.add_argument('--batch_size', '-b', type=int, default=1000)
args = parser.parse_args()


logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
logdir = os.path.join('data', logdir)

print("------")
print(logdir)
print("------")

max_path_length = args.ep_len if args.ep_len > 0 else None


for data_gen in range(args.data_gen[0], args.data_gen[1], args.data_gen[2]):
    for w_radar in range(args.w_radar[0], args.w_radar[1], args.w_radar[2]):
        for pv in range(args.pv[0], args.pv[1], args.pv[2]):
            for rd_bad2bad in np.arange(args.rd_bad2bad[0],args.rd_bad2bad[1],args.rd_bad2bad[2]):
            
                for e in range(args.n_experiments):
                    """ Set random seeds 
                    https://keras.io/getting_started/faq/
                    """
                    seed = args.seed + e*10
                    # The below is necessary for starting Numpy and Python generated random numbers in a well-defined initial state.
                    np.random.seed(seed)
                    python_random.seed(seed)
                    
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
                                  }
                    
                    logdir_w_params = logdir + "_rdbad_{}_wr{}_gen{}".format(rd_bad2bad, w_radar, data_gen)
                    
                    test(
                        exp_name = args.exp_name,
                        env_name = args.env_name,
                        env_config = env_config,
                        n_iter = args.n_iter, 
                        min_timesteps_per_batch = args.batch_size, 
                        max_path_length = max_path_length,
                        seed = args.seed,
                        mode = args.mode,
                        CW = args.CW,
                        logdir = os.path.join(logdir_w_params,'%d'%seed),
                        )
                    