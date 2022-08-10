# Multi-Agent JRC-AoI Programs
[![DOI](https://zenodo.org/badge/422615600.svg)](https://zenodo.org/badge/latestdoi/422615600)

Supplementary material for the following papers:

J. Lee, D. Niyato, Y. L. Guan and D. I. Kim, "Learning to Schedule Joint Radar-Communication Requests for Optimal Information Freshness," 2021 IEEE Intelligent Vehicles Symposium (IV), 2021, pp. 8-15, doi: 10.1109/IV48863.2021.9575131.

J. Lee, T. D. Niyato, Y. L. Guan and D. I. Kim, "Learning to Schedule Joint Radar-Communication with Deep Multi-Agent Reinforcement Learning," in IEEE Transactions on Vehicular Technology, doi: 10.1109/TVT.2021.3124810.

## Getting started
Install the dependencies listed in 'requirements.txt'.

## Running Experiments
The multi-agent PPO training process, multi-agent A2C training process, and the binary exponential backoff (BEB) baseline algorithm, may be run from the command line. Examples are provided below.
Multi-agent PPO:
```
python trainPPO_JRC.py JRCAoI_multi --num_users 4 --num_agents 1 --rd_bad2b 1.0 1.1 0.1 --w_radar 5 6 1 --w_ovf 1 --w_age 0.002 --data_gen 1 2 1 -s 64 64 -sc 64 64 -dc -ep 400 -n 1000 -b 4000 --ex_time -e 2 --seed 31 -lr 0.0001 --entrop_loss_coef 0.01 --exp_name PPO_4usr1agnt_entrop_coef0.01_lr1e-4
```
Multi-agent A2C:
```
python trainA2C_JRC.py JRCAoI_multi --num_users 4 --num_agents 1 --rd_bad2b 1.0 1.1 0.1 --w_radar 5 6 1 --w_ovf 1 --w_age 0.002 --data_gen 1 2 1 -s 64 64 -sc 64 64 -dc -ep 400 -n 1000 -b 4000 --ex_time -e 5 -lr 0.0001 --entrop_loss_coef 0.01 --exp_name A2C_4usr1agnt_entrop_coef0.01_lr1e-4
```
Binary Exponential Backoff:
```
python test_JRC_multi.py JRCAoI_multi --num_users 4 --mode csma-ca --rd_bad2b 1.0 1.1 0.1 --w_radar 5 6 1 --w_ovf 1 --w_age 0.002 --data_gen 1 2 1 -ep 400 -n 1000 -b 4000 -e 5 --CW 2 16 --exp_name Multi_v0_csma-ca_4usr_CWmax16
```
