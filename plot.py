import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import numpy as np

"""


"""

def plot_data(data, value="AverageCost"):
    sns.set(style="whitegrid", font_scale=1.1)
    myplot = sns.lineplot(x=data["Iteration"], y=data[value], hue=data[""])
    
    """ Min/Max line and annotations """
    x = np.array(data.groupby('').max()['Iteration'])
    y = np.array(data.groupby('').max()[value])
    text = 'max'
    xy = (x/2,y)
    
    axes = myplot.axes
    axes.set_xlim(0,1000)
    
    plt.legend(bbox_to_anchor=(-0, 1.05), loc='lower left', borderaxespad=0.,ncol=2)
    plt.show()


def get_datasets(fpath, condition=None):
    unit = 0
    datasets = []
    for root, dir, files in os.walk(fpath):
        if 'log.txt' in files:
            param_path = open(os.path.join(root,'params.json'))
            params = json.load(param_path)
            exp_name = params['exp_name']
            
            log_path = os.path.join(root,'log.txt')
            experiment_data = pd.read_table(log_path)

            experiment_data.insert(
                len(experiment_data.columns),
                'Unit',
                unit
                )        
            experiment_data.insert(
                len(experiment_data.columns),
                '',#'Condition'
                condition or exp_name
                )
                
            datasets.append(experiment_data)
            unit += 1

    return datasets


def main():
    class DotDict(dict):
        def __init__(self, **kwds):
            self.update(kwds)
            self.__dict__ = self
    
    args = DotDict()
    
    args.logdir_unif_rand = (
        'data/Multi_v0_unif_rand_2usr_bs4000_JRCAoI_multi_v0_11-03-2021_14-43-12_rdbad_1.0_wr10_gen1',
        )
    
    args.logdir_PPO = (
        'data/Multi_v0_PPO2.6_2usr1agnt_acnn6464_bs4000_JRCAoI_multi_v0a_11-03-2021_18-33-38_rdbad_1.0_wr5_gen1.0_nn[64, 64]',       # v0a: shared parameters, simultaneous ac collision
        )
    
    args.logdir_PPO_tuning = (
        'data/Multi_v0_PPO2.6_2usr1agnt_entrop_loss0.01_lr1e-4_JRCAoI_multi_v0a_22-03-2021_12-15-58_rdbad_1.0_wr5_gen1.0_nn[64, 64]',   # postitive coefficient
        'data/Multi_v0_PPO2.6_2usr1agnt_entrop_loss0.05_lr1e-4_JRCAoI_multi_v0a_22-03-2021_12-16-02_rdbad_1.0_wr5_gen1.0_nn[64, 64]',
        )
    
    """ Binary exponential backoff """
    args.logdir_CSMA_BEB = (
        'data/Multi_v0_csma-ca_4usr_CWmax4_JRCAoI_multi_v0d2_21-04-2021_14-08-48_rdbad_1.0_wr5_gen1',
        'data/Multi_v0_csma-ca_4usr_CWmax16_JRCAoI_multi_v0d2_21-04-2021_14-28-36_rdbad_1.0_wr5_gen1',
        )
    
    args.logdir_CSMA_A2C = (
        'data/A2C2.6b_4usr1agnt_entrop_coef0.01_lr1e-4_JRCAoI_multi_v0d2_20-04-2021_16-57-21_rdbad_1.0_wr5_gen1.0_nn[64, 64]',
        )
    
    """ PPO on CSMA-CA environment v0d2 """
    args.logdir_CSMA_PPO = (
        'data/PPO2.6b_4usr1agnt_entrop_coef0.01_lr1e-4_JRCAoI_multi_v0d2_14-04-2021_19-58-58_rdbad_1.0_wr5_gen1.0_nn[64, 64]',          # v2.6b: correction to loss calc
        'data/PPO2.6bconv2_4usr1agnt_entrop_coef0.01_lr1e-4_JRCAoI_multi_v0d2_15-04-2021_22-56-47_rdbad_1.0_wr5_gen1.0_nn[64, 64]',     # conv, 10 filters
        'data/PPO2.6bconv2_4usr1agnt_entrop_coef_const0.01_lr1e-4_JRCAoI_multi_v0d2_21-04-2021_15-21-01_rdbad_1.0_wr5_gen1.0_nn[64, 64]',
        'data/PPO2.6bconv2_4usr1agnt_entrop_coef0_lr1e-4_JRCAoI_multi_v0d2_21-04-2021_15-19-28_rdbad_1.0_wr5_gen1.0_nn[64, 64]',
        )
    
    args.logdir = args.logdir_unif_rand + args.logdir_PPO + args.logdir_PPO_tuning
    # args.logdir = args.logdir_unif_rand + args.logdir_PPO_tuning + args.logdir_PPO_CSMA
    args.logdir = args.logdir_CSMA_BEB +  args.logdir_CSMA_A2C + args.logdir_CSMA_PPO
    # args.logdir = args.logdir_CSMA_A2C + args.logdir_CSMA_PPO
    # args.logdir = args.logdir_CSMA_PPO
    # args.logdir = args.logdir_CSMA_BEB
    
    """ Legend """
    args.legend = ['BEB, $CW_{max}=4$','BEB, $CW_{max}=16$','A2C','PPO - ent coef no decay','PPO - MLP','PPO - CNN']
    args.legend = ['BEB - $CW_{max}=4$','BEB - $CW_{max}=16$','A2C - MLP','PPO - MLP','PPO - CNN','PPO - CNN, const $k_2$', 'PPO - CNN, $k_2=0$']
    args.value = ['Average Reward']
    # args.value = ['Entropy Bonus 1']
    # args.value = ['Average Reward', 'Reward1','Entropy Bonus 1']
    # args.value = ['Average Reward', 'Reward1','Entropy Bonus 1','radar action %','Throughput 1','comm action req %']
    # args.value = ['Average Reward', 'Reward1', 'Reward2', 'r_age', 'r_radar', 'r_overflow', 'throughput']
    
    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True

    data = []
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            data += get_datasets(logdir, legend_title)
    else:
        for logdir in args.logdir:
            data += get_datasets(logdir)

    if isinstance(args.value, list):
        values = args.value
    else:
        values = [args.value]
    
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    
    data.rename(columns={'Entropy Loss 1': 'Entropy Bonus 1',
                         'throughput': 'Throughput 1'}
                ,inplace=True)
    for value in values:
        plot_data(data, value=value)

if __name__ == "__main__":
    main()
