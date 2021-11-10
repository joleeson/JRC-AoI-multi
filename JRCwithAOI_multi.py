import gym
import numpy as np
import random
from gym import spaces
from gym.utils import seeding
from config_jrc_aoi_multi import transition_probability, unexpected_ev_prob, state_space_size, r_params


"""
Code for the paper "Learning to Schedule Joint Radar-Communication with Deep
Multi-Agent Reinforcement Learning", IEEE Transactions on Vehicular Technology

Author: Joash Lee

MDP for JRC-Aoi Problem
 
"""
class AV_Environment(gym.Env):
    def __init__(self, env_config):
        
        self.N = env_config["num_users"]          # No. of users
        r_params['age_obj'] = env_config['age_obj']
        r_params['w_overflow'] = env_config['w_overflow']
        r_params['phi'] = env_config['phi']
        assert (r_params['age_obj'] == 'peak' or r_params['age_obj'] =='avg')
        self.opt_obj = r_params['age_obj']
        self.max_age =  r_params['A_max']
        urg_lvls = transition_probability['arrival_mean'].shape[0]
        high = np.concatenate(
            ((self.max_age+1)*np.ones(state_space_size['data_size']),        # age of each packet
             urg_lvls*np.ones(state_space_size['data_size']),           # urgency level of each packet
             np.ones(state_space_size['channel_size'] +
                     state_space_size['road_size'] +                    # environmental conditions
                     state_space_size['weather_size'] + 
                     state_space_size['speed_size'] + 
                     state_space_size['object_size']),
             (self.max_age+1)*np.ones(state_space_size['num_classes']),      # age of each class
            np.ones(state_space_size['user_index']),
            np.ones(state_space_size['time'])*200,
             )
            )
        self.observation_space = spaces.Box(
            low=0, high=high, shape=(state_space_size['data_size']*2 +
            state_space_size['channel_size'] +
            state_space_size['road_size'] +
            state_space_size['weather_size'] +
            state_space_size['speed_size'] +
            state_space_size['object_size'] +
            state_space_size['num_classes'] +
            state_space_size['user_index'] +
            state_space_size['time'],)
            )
        self.extra_ob_dim = 2     # user specific observation dimensions
        
        # choice of idle, radar mode or which data class to transmit
        self.action_space = spaces.Discrete(urg_lvls + 2)
        
        unexpected_ev_prob['occur_with_fast_speed'] = env_config['pv']
        r_params['w_radar'] = env_config['w_radar']
        transition_probability['arrival_mean'] = env_config['data_gen']/10 * np.ones((state_space_size['num_classes'],))
        transition_probability['road_sw_bad_to_bad'] = env_config['road_sw_bad_to_bad']
        
        self.seed(123)
        # self.state = self.reset()
        # print(self.state)
    
    def action_decode(self, action):
        index = action - 1
        ac = np.unravel_index(index, (state_space_size['data_size'],state_space_size['data_size']))
        return ac
    
    def markov_transition(self, current_state, state_id):
        """
        Parameters
        ----------
        current_state : indicator state for one of the following variables: channel, road, weather, speed
        state_id : indicator of which state variable is being input as 'current_state'

        Returns
        -------
        current_state : indicator state at time t+1

        """
        if current_state > 1 or current_state < 0:
            raise Exception('Invalid current_state')
        if state_id > 5 or state_id < 1:
            raise Exception('state_id should not exceed 5 or below 1')

        markov_probability = 0.0
        if current_state == 1:
            if state_id == 1:
                markov_probability = transition_probability['channel_sw_bad_to_bad']
            if state_id == 2:
                markov_probability = transition_probability['road_sw_bad_to_bad']
            if state_id == 3:
                markov_probability = transition_probability['weather_sw_bad_to_bad']
            if state_id == 4:
                markov_probability = transition_probability['speed_sw_fast_to_fast']
            if state_id == 5:
                markov_probability = transition_probability['object_sw_moving_to_moving']
            if self.nprandom.uniform() < markov_probability:
                current_state = 1
            else:
                current_state = 0
        else:
            if state_id == 1:
                markov_probability = transition_probability['channel_sw_good_to_good']
            if state_id == 2:
                markov_probability = transition_probability['road_sw_good_to_good']
            if state_id == 3:
                markov_probability = transition_probability['weather_sw_good_to_good']
            if state_id == 4:
                markov_probability = transition_probability['speed_sw_slow_to_slow']
            if state_id == 5:
                markov_probability = transition_probability['object_sw_static_to_static']
            if self.nprandom.uniform() < markov_probability:
                current_state = 0
            else:
                current_state = 1

        return current_state
    
    
    def state_transition(self, state, action, transmitted_packets, new_packets, n):
        """
        The state transition function. 
        Called by:  'step()' function.
        Used by:    'markov_transition'
        
        Parameters
        ----------
        state : Markov state at time t
        action : action at time t

        Returns
        -------
        Markov state at time t+1

        """
        
        self.data_age[str(n+1)][self.data_urg[str(n+1)]>0] = self.data_age[str(n+1)][self.data_urg[str(n+1)]>0] + 1   # Increment age counter by 1
        self.class_age[str(n+1)] = self.class_age[str(n+1)] + 1        
                
        new_data_urg = np.concatenate([np.tile(i+1,new_packets[i]) for i in range(state_space_size['num_classes'])])
        self.data_age[str(n+1)] = np.concatenate((self.data_age[str(n+1)][self.data_urg[str(n+1)]!=0], np.zeros(state_space_size['data_size'])[self.data_urg[str(n+1)]==0]))
        self.data_urg[str(n+1)] = np.concatenate((self.data_urg[str(n+1)][self.data_urg[str(n+1)]!=0], new_data_urg, self.data_urg[str(n+1)][self.data_urg[str(n+1)]==0]))[:state_space_size['data_size']]
        
        # assert self.data_age.shape == (11,)
        # assert self.data_urg.shape == (11,)
        
        # TODO transition for Markov chain
        # self.channel[str(n+1)] = np.array([self.markov_transition(self.channel[str(n+1)], 1)])
        self.road[str(n+1)] = np.array([self.markov_transition(self.road[str(n+1)], 2)])
        self.weather[str(n+1)] = np.array([self.markov_transition(self.weather[str(n+1)], 3)])
        if self.object[str(n+1)] == 1:
            if action == (self.action_space.n-1):  # if RADAR mode
                self.speed[str(n+1)] = np.array([0], dtype=int)
                # if the AV's speed is slowed down, the moving object's state should be changed as well
                self.object[str(n+1)] =  np.array([0], dtype=int) if np.random.uniform() < 0.9 else  np.array([1], dtype=int)  # 90% it change to good state
            else:
                self.speed[str(n+1)] = np.array([self.markov_transition(self.speed[str(n+1)], 4)], dtype=int)
                self.object[str(n+1)] = np.array([self.markov_transition(self.object[str(n+1)], 5)], dtype=int)
        else:
            self.speed[str(n+1)] = np.array([self.markov_transition(self.speed[str(n+1)], 4)], dtype=int)
            self.object[str(n+1)] = np.array([self.markov_transition(self.object[str(n+1)], 5)], dtype=int)
        
        # self.t_last[str(n+1)] = self.t_last[str(n+1)] + 1 if action==0 else np.array([0], dtype=int)      # increment counter if no-op is taken, else reset counter to zero
        # self.t_last[str(n+1)] = self.t_last[str(n+1)] + 1 if num_transmitted_packets==0 else np.array([0], dtype=int)      # increment counter if no packets sent, else reset counter to zero
        # Log statistics
        if n==0:
            self.episode_observation['throughput'] += int(np.sum(transmitted_packets))
            self.episode_observation['data_counter'] += int(np.sum(new_packets))
            self.episode_observation['urgency_counter'] += int(np.sum(new_data_urg))
        
        return self._get_obs(n)

    def risk_assessment(self, state, n):
        
        nb_bad_bits = np.sum((self.road[str(n+1)], self.weather[str(n+1)], self.speed[str(n+1)], self.object[str(n+1)]))
        
        if self.nprandom.uniform() < np.exp(nb_bad_bits)/np.exp(4):
            unexpected_event = 1
        else:
            unexpected_event = 0
        
        return unexpected_event

    def get_reward(self, state, action, n):
        """
        The environment generates a reward rt when the agent makes an action a(t) from state s(t)

        Parameters
        ----------
        state : state at time t
        action : action taken by the agent at time t

        Returns
        -------
        reward : reward at time t
        transmitted_packets: (11,) np array where 1s indicate transmission and 0 indicates no transmission

        """
        
        unexpected_ev_occurs = self.risk_assessment(state, n)
        # channel_state = self.channel
        nb_bad_bits = np.sum((self.road[str(n+1)], self.weather[str(n+1)], self.speed[str(n+1)], self.object[str(n+1)]))
        r_age, r_radar, r_overflow = 0, 0, 0
        r_age_by_channel = np.zeros((state_space_size['num_classes']), dtype=int)
        
        expired_data = np.ones_like(self.data_age[str(n+1)], dtype=int) * (self.data_age[str(n+1)] > r_params['A_max'])
        # if np.sum(expired_data) > 0:
        #     print('expired data')
        
        transmitted_packets = np.zeros_like(self.data_urg[str(n+1)], dtype=int)
        
        if n == 0:
            self.episode_observation['state_map'][nb_bad_bits] += 1
            # if max(self.class_age) > 90:
            #     print('max class age is above 90')
            self.episode_observation['state_age_map'][np.arange(state_space_size['num_classes']), np.minimum(np.ones_like(self.class_age[str(n+1)])*self.max_age,self.class_age[str(n+1)].astype(int))] += 1
        
        if action in np.arange(state_space_size['num_classes'])+1:                  # COMMUNICATION Mode
            if n == 0:
                self.episode_observation['comm_counter'] += 1
                self.episode_observation['action_age_map'][int(action-1), min(self.max_age,self.class_age[str(n+1)].astype(int)[int(action-1)])] += 1
            
            if unexpected_ev_occurs == 0:   # No bad/unexpected events
                idx = np.argwhere(self.data_urg[str(n+1)] == action).squeeze(1)
                
                transmitted_packets = transmitted_packets + np.eye(state_space_size['data_size'])[idx[0]] if len(idx) != 0 else transmitted_packets
                if (len(idx) != 0):
                    self.class_age[str(n+1)][action-1] = self.data_age[str(n+1)][idx[0]]       # if data was transmitted, set age of channel to age of last transmitted packet
        
        if (action != (self.action_space.n-1)) and unexpected_ev_occurs: # if unexpected event occurs and radar not used
            r_radar = -nb_bad_bits
            self.episode_observation['wrong_mode_actions'] += 1 if n == 0 else 0
        
        if action == (self.action_space.n-1) and n==0:                               # RADAR Mode
            self.episode_observation['action_map'][nb_bad_bits] += 1 if n == 0 else 0
            self.episode_observation['radar_counter'] += 1
        
        # peak_age = int(np.sum((np.maximum(transmitted_packets,expired_data) * self.data_age[str(n+1)])))
        peak_age = int(np.sum((transmitted_packets * self.data_age[str(n+1)])))
        # if peak_age > r_params['A_max']:
        #     print('Peak age > A_max')
        # Remove transmitted packets and expired packets from the data queue
        unexpired_data = (self.data_age[str(n+1)] <= r_params['A_max'])
        self.data_urg[str(n+1)] = (transmitted_packets==0) * (unexpired_data) * self.data_urg[str(n+1)]
        self.data_age[str(n+1)] = (transmitted_packets==0) * (unexpired_data) * self.data_age[str(n+1)]
        
        r_class_age = np.sum(-(self.class_age[str(n+1)]+1) * (np.arange(state_space_size['num_classes'])+1))    # multiply class age by urgency level
        
        
        new_packets = self.nprandom.poisson(transition_probability['arrival_mean'])     # Num new packets arriving in data queue.    
        excess_packets = np.sum(self.data_urg[str(n+1)] > 0 - expired_data - transmitted_packets) + np.sum(new_packets) - state_space_size['data_size']
        excess_packets = excess_packets * (excess_packets > 0)
        r_overflow += - excess_packets
        
        # reward = r_params['w_age']*r_age + r_params['w_radar']*r_radar + r_params['w_overflow']*r_overflow
        reward = r_params['w_age']*r_class_age + r_params['w_radar']*r_radar + r_params['w_overflow']*r_overflow
        
        if type(reward) == type(np.array([1])):
            print('reward is np')
        
        # Log statistics
        if n==0:
            if unexpected_ev_occurs == 1:
                self.episode_observation['unexpected_ev_counter'] += 1
            # eliminated = np.maximum(transmitted_packets,expired_data)
            self.episode_observation['peak_age_counter'] = self.episode_observation['peak_age_counter'] + peak_age
            # self.episode_observation['r_age'] += int(r_age)
            self.episode_observation['r_radar'] += int(r_radar)
            self.episode_observation['r_overflow'] += int(r_overflow)
            for i in range(state_space_size['num_classes']):
                # self.episode_observation['r_age'+str(i+1)] += int(r_age_by_channel[i])
                self.episode_observation['r_class_age'+str(i+1)] += int(self.class_age[str(n+1)][i] * (i+1))
            self.episode_observation['r_class_age'] += int(r_class_age)
            
        return reward, transmitted_packets, new_packets
    
    
    def step(self, action):
        """
        The agent takes a step in the environment.
        Parameters
        ----------
        action : action the agent has decided to take at time t

        Returns
        -------
        next_state : state at time t+1
        reward : reward at time t
        done : indicator of wether the episode has ended

        """
        actions = np.empty((0),dtype=int)
        priorities = np.empty((0),dtype=int)
        for n in range(self.N):
            assert self.action_space.contains(action[str(n+1)]), "%r (%s) invalid" % (action[str(n+1)], type(action[str(n+1)]))
            actions = np.append(actions, action[str(n+1)])
            priorities = np.concatenate((priorities, self.channel[str(n+1)].reshape(-1)))
        
        if actions[0] in np.arange(state_space_size['num_classes'])+1:
            self.episode_observation['comm_req_counter'] += 1
        elif actions[0] == (self.action_space.n-1):
            self.episode_observation['radar_req_counter'] += 1
        
        if np.sum(actions>0) > 1:
            if np.sum(priorities * (actions>0)) == 1:
                actions = (priorities == 1) * actions
            else:
                idxs = np.argwhere(actions>0).reshape(-1)
                idx = np.random.randint(len(idxs))
                action = actions[idx]
                actions = np.zeros_like(actions)
                actions[idx] = action
        self.t_id = self.t_id + 1 if (np.sum(actions) == 0) else np.array([0])         # Increment channel idle counter if all agents take no action
        self.cycle_count = self.cycle_count + 1 if self.cycle_count < self.N else np.array([1])
        
        next_state = (self.state).copy()
        
        reward = {}
        for n in range(self.N):
            reward[str(n+1)], transmitted_packets, new_packets = self.get_reward(next_state, actions[n], n)
            if (np.sum(actions) == 0):                              # If medium is idle
                self.t_last[str(n+1)] = self.t_last[str(n+1)] + 1
            elif actions[n] != 0:                                   # If medium is not used by agent in the current time step
                self.t_last[str(n+1)] = np.array([0])
            self.channel[str(n+1)] = np.array([1], dtype=int) if np.sum(transmitted_packets)>0 else np.array([0], dtype=int)
            next_state[str(n+1)] = self.state_transition(next_state, actions[n], transmitted_packets, new_packets, n)
            
        self.state = next_state 
        
        self.episode_observation['step_counter'] += 1
        if self.episode_observation['step_counter'] == 400:
            done = True
            # print("Throughput: ", self.episode_observation['throughput'])
            print("Peak age: ", self.episode_observation['peak_age_counter']/(self.episode_observation['throughput'] + 1e-7))
            # print("Wrong mode actions: ", self.episode_observation['wrong_mode_actions'])
        else:
            done = False
        
        return self.state, reward, done, {}

    def reset(self):
        """
        Reset the environment and return the initial state s0

        Returns
        -------
        state at time t=0

        """
        num_actions = state_space_size['num_classes'] + 2
        self.episode_observation = {
            'step_counter': 0,
            'unexpected_ev_counter': 0,
            'wrong_mode_actions': 0,
            'throughput': 0,
            'sim_reward': 0,
            'data_counter': 0,
            'urgency_counter': 0,
            'peak_age_counter': 0,
            'good_ch_comm': 0,
            'comm_counter': 0,
            'radar_counter': 0,
            'comm_req_counter': 0,
            'radar_req_counter': 0,
            'r_age': 0,
            'r_radar': 0,
            'r_class_age': 0,
            'action_map': np.zeros((5,), dtype=int),  # num bad bits x channel. Counter for discrete category is incremented when comm action is taken
            'state_map': np.zeros((5,), dtype=int),   # num bad bits x channel. Counter for discrete category is incremented when state is visited
            'action_age_map': np.zeros((num_actions,r_params['A_max']+1), dtype=int),  # num classes x age. Counter for discrete category is incremented when comm action is taken
            'state_age_map': np.zeros((num_actions,r_params['A_max']+1), dtype=int),   # num classes x age. Counter for discrete category is incremented when state is visited
            'r_overflow': 0,
        }
        for i in range(state_space_size['num_classes']):
            self.episode_observation['r_age'+str(i+1)] = 0
            self.episode_observation['r_class_age'+str(i+1)] = 0
        
        self.t_id = np.array([0], dtype=int)            # Num steps that channel is idle
        self.cycle_count = np.array([1], dtype=int)
        
        state, self.data_age, self.data_urg, self.channel, self.road, self.weather, self.speed, self.object, self.class_age, self.t_last = {},{},{},{},{},{},{},{},{},{}
        
        for n in range(self.N):
            self.data_urg[str(n+1)] = np.zeros(state_space_size['data_size'], dtype=int)
            self.data_age[str(n+1)] = np.zeros(state_space_size['data_size'], dtype=int)
            
            self.class_age[str(n+1)] = np.zeros(state_space_size['num_classes'], dtype=int)
            
            new_packets = self.nprandom.poisson(transition_probability['arrival_mean'])
            new_data_urg = np.concatenate([np.tile(i+1,new_packets[i]) for i in range(state_space_size['num_classes'])])
            len_diff = state_space_size['data_size']-len(new_data_urg)
            if len_diff >= 0:
                self.data_urg[str(n+1)] = np.pad(new_data_urg, (0,len_diff),'constant',constant_values=0)
            else:
                self.data_urg[str(n+1)] = new_data_urg[:state_space_size['data_size']]
            
            self.channel[str(n+1)] = np.array([0], dtype=int)
            self.road[str(n+1)] = np.random.randint(2,size=state_space_size['road_size'])
            self.weather[str(n+1)] = np.random.randint(2,size=state_space_size['weather_size'])
            self.speed[str(n+1)] = np.random.randint(2,size=state_space_size['speed_size'])
            self.object[str(n+1)] = np.random.randint(2,size=state_space_size['object_size'])
            
            self.t_last[str(n+1)] = np.array([0], dtype=int)        # Num steps since user (n+1) last transmitted
            
            # Log statistics
            if n == 0:
                self.episode_observation['data_counter'] += int(np.sum(new_packets))
                self.episode_observation['urgency_counter'] += int(np.sum(new_data_urg))
        
        user_with_priority = np.random.randint(self.N)
        self.channel[str(user_with_priority)] = np.array([1], dtype=int)
        
        for n in range(self.N):
            state[str(n+1)] = self._get_obs(n)
        # assert state.shape == (27,)
        
        self.state = state.copy()
        return state
    
    def _get_obs(self, n):
        ID = np.array([n+1], dtype=int)
        # feature for parameter sharing network: ID/self.N
        # state = np.concatenate((self.data_age[str(n+1)]/self.max_age, self.data_urg[str(n+1)]/state_space_size['num_classes'], self.road[str(n+1)], self.weather[str(n+1)], 
        #                         self.speed[str(n+1)], self.object[str(n+1)], self.class_age[str(n+1)]/self.max_age, self.cycle_count/self.N, self.t_id/10))
        
        # Non-user specific features are at the end of the state vector
        state = np.concatenate((self.data_age[str(n+1)]/self.max_age, self.data_urg[str(n+1)]/state_space_size['num_classes'],
                                self.channel[str(n+1)], self.road[str(n+1)], self.weather[str(n+1)], 
                                self.speed[str(n+1)], self.object[str(n+1)], self.class_age[str(n+1)]/self.max_age,
                                self.t_last[str(n+1)]/10, ID/self.N, self.cycle_count/self.N, self.t_id/10))
        state = np.expand_dims(state, axis=0)
        # assert state.shape == (27,)
        return state
    
    def seed(self, seed=None):
        self.nprandom, seed = seeding.np_random(seed)
        return [seed]
    
    def seed_new(self, seed):
        np.random.seed = seed
    
    def is_terminated(self):
        ...

