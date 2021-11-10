
from __future__ import division
import numpy as np

"""
 System parameters for the MDP environment in 'JRCwithAOI_multi.py'
"""
UNEXPECTED_EV_PROB = 0.05
transition_probability = {
    'channel_sw_bad_to_bad': 0.1,  # variable 2
    'channel_sw_good_to_good': 0.9,
    'road_sw_bad_to_bad': 0.1,
    'road_sw_good_to_good': 0.9,
    'weather_sw_bad_to_bad': 0.1,
    'weather_sw_good_to_good': 0.9,
    'speed_sw_fast_to_fast': 0.1,
    'speed_sw_slow_to_slow': 0.9,
    'object_sw_moving_to_moving': 0.1,
    'object_sw_static_to_static': 0.9,
    'arrival_mean': np.array([0.2,0.2,0.2]),
}

unexpected_ev_prob = {
    'occur_with_bad_road': UNEXPECTED_EV_PROB,
    'occur_with_good_road': UNEXPECTED_EV_PROB / 10,
    'occur_with_bad_weather': UNEXPECTED_EV_PROB,
    'occur_with_good_weather': UNEXPECTED_EV_PROB / 10,
    'occur_with_fast_speed': UNEXPECTED_EV_PROB * 2, #2,  # variable 1
    'occur_with_slow_speed': UNEXPECTED_EV_PROB / 10,
    'occur_with_moving_object': UNEXPECTED_EV_PROB,
    'occur_with_static_object': UNEXPECTED_EV_PROB / 10,
}

state_space_size = {
    'data_size': 11,
    'channel_size': 1,
    'road_size': 1,
    'weather_size': 1,
    'speed_size': 1,
    'object_size': 1,
    'num_classes': 3,
    'time': 3,
    'user_index': 1,
}

action_space_size = {
    'action_size': 2,
}

r_params = {
    'A_max': 100, #20
    'phi': 1000,
    'w_overflow': 0, #1,
    'w_age': 0.002, #0.01, #1
    'w_comm': 0, #1,
    'w_radar': 10, #5, #1,
    'age_obj': 'peak',
    }


#  Parameters for testing DQN agent
test_parameters = {
    'test_id': 2,
    'nb_steps': 2500 * 400,
    'nb_epsilon_linear': 500 * 400,
    'target_model_update': 1e-3,
    'gamma': 0.99,
    'alpha': 0.01,
    'add_simulation': False,
}

"""
'test_id': 15,
'occur_with_fast_speed': UNEXPECTED_EV_PROB,
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 24,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 2,
'occur_with_moving_object': UNEXPECTED_EV_PROB * 2,
--------------------------
'test_id': 16,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 4,
'occur_with_moving_object': UNEXPECTED_EV_PROB * 4,
--------------------------
'test_id': 25,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 8,
'occur_with_moving_object': UNEXPECTED_EV_PROB * 8,
--------------------------
'test_id': 17,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 16,
'occur_with_moving_object': UNEXPECTED_EV_PROB * 16,
--------------------------
--------------------------
'test_id': 26,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 2, # 0.1
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 27,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 4, # 0.2
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 28,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 6, # 0.3
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 29,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 8, # 0.4
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 30,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 10, # 0.5
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 31,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 12, # 0.6
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 32,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 14, # 0.7
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 33,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 16, # 0.8
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 34,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 18, # 0.9
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
'test_id': 35,
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 20, # 1.0
'occur_with_moving_object': UNEXPECTED_EV_PROB,
--------------------------
--------------------------
'test_id': 36, 81
'channel_sw_bad_to_bad': 0.1 * 1,  # 0.1
--------------------------
'test_id': 37, 82
'channel_sw_bad_to_bad': 0.1 * 2,  # 0.2
--------------------------
'test_id': 38, 83
'channel_sw_bad_to_bad': 0.1 * 3,  # 0.3
--------------------------
'test_id': 39, 84
'channel_sw_bad_to_bad': 0.1 * 4,  # 0.4
--------------------------
'test_id': 40, 85
'channel_sw_bad_to_bad': 0.1 * 5,  # 0.5
--------------------------
'test_id': 41, 86
'channel_sw_bad_to_bad': 0.1 * 6,  # 0.6
--------------------------
'test_id': 42, 87
'channel_sw_bad_to_bad': 0.1 * 7,  # 0.7
--------------------------
'test_id': 43, 88
'channel_sw_bad_to_bad': 0.1 * 8,  # 0.8
--------------------------
'test_id': 44, 89
'channel_sw_bad_to_bad': 0.1 * 9,  # 0.9
--------------------------
'test_id': 45, 90
'channel_sw_bad_to_bad': 0.1 * 10,  # 10
--------------------------
--------------------------
'test_id': 46, 71
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 2, # 0.1
--------------------------
'test_id': 47, 72
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 4, # 0.2
-------------------------
'test_id': 48, 73
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 6, # 0.3
--------------------------
'test_id': 49, 74
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 8, # 0.4
--------------------------
'test_id': 50, 75
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 10, # 0.5
--------------------------
'test_id': 51, 76
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 12, # 0.6
--------------------------
'test_id': 52, 77
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 14, # 0.7
--------------------------
'test_id': 53, 78
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 16, # 0.8
--------------------------
'test_id': 54, 79
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 18, # 0.9
--------------------------
'test_id': 55, 80
'occur_with_fast_speed': UNEXPECTED_EV_PROB * 20, # 1.0
--------------------------
--------------------------

'eps_min': 0.1 : 102
--------------------------
'eps_min': 0.05 : 103
--------------------------
'eps_min': 0.1 : 104
--------------------------
'eps_min': 0.1 : 105, tanh(r_t)
--------------------------
'eps_min': 0.1 : 106
--------------------------


"""


