import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind_from_stats
from tbparse import SummaryReader
from matplotlib import rc
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{color}')


def aggregate_results(run_generic_name, runs_directory='./', eval_files=None, run_name_exclude=None, os_list=None, metric='EM'):
    """
    @param run_generic_name: ex. gpt2-medium-seed
    @return:
    """
    assert metric in ['EM', 'F1']
    if os_list is None:
        os_list = os.listdir(runs_directory)
    extracted_runs_names = [name for name in os_list
                            if name.startswith(run_generic_name)]
    if run_name_exclude:
        extracted_runs_names = [
            name for name in extracted_runs_names if run_name_exclude not in name]
    print(f'Aggregating from {len(extracted_runs_names)} runs')
    # for i, name in enumerate(extracted_runs_names):
    #     print(f'{i+1}) {name}')

    if eval_files is None:
        eval_files = ['eval_d1consis', 'eval_d2consis', 'eval_no_qd_baseline']

    all_results = []
    for name in extracted_runs_names:
        # seed = int(name[name.find('B-s') + 3:])
        run_results = []
        for eval_file in eval_files:
            try:
                with open(os.path.join(runs_directory, name, eval_file + '_results.json')) as f:
                    data = json.load(f)
            except FileNotFoundError:
                # print(f'File {eval_file} not found in {name}')
                break
            # except Exception:
            #     print('Broken json', seed)
            #     continue

            run_results.append(data[f'{metric} ' + '{k}'])
        if len(run_results) == len(eval_files):
            all_results.append(run_results)
    assert len(all_results) > 0
    print(f'Successfully loaded full results from {len(all_results)} runs')

    averaged = np.array(all_results).mean(axis=0)
    # ddof=1 for unbiased std (bessel's correction)
    stds = np.array(all_results).std(axis=0, ddof=1)
    res_dict = dict(
        zip(eval_files, zip(averaged, stds, [len(all_results)]*len(eval_files))))

    for k in dict(res_dict):
        if k.startswith('eval_'):
            res_dict[k[5:]] = res_dict.pop(k)

    df = pd.DataFrame.from_dict(res_dict, orient='index', columns=[
                                f'{metric} avg', f'{metric} std', 'n_runs'])
    df = df.drop(columns=['n_runs'])
    print(df)
    return res_dict


def ttest_res_dict(res_dict, var1, var2):
    return ttest_ind_from_stats(mean1=res_dict[var1][0], std1=res_dict[var1][1], nobs1=res_dict[var1][2],
                                mean2=res_dict[var2][0], std2=res_dict[var2][1], nobs2=res_dict[var2][2],
                                alternative='greater')


def prettify_labels(labels_list, labels_mapping=None):
    if labels_mapping is None:
        labels_mapping = {'d1consis': r'${{QA}^-, {D}_{1,consis}^+}$',
                          'd2consis': r'${{QA}^-, {D}_{2,consis}^+}$',
                          'qd2inconsis': r'{{QA}^+, {D}_{{2,incons}}^+}$',
                          }
        
    return [labels_mapping.get(label, label) for label in labels_list]
    
    
def make_experiment_plot(exp_name, stage_paths, thruncate_stages_after_epoch=None, eval_each_epochs_per_stage=None,
                         tags=['eval/d1consis_EM', 'eval/d2consis_EM'], os_list=None, ylabel='Value', figsize=(8,4)):
    """
    exp_name - name of the experiment (top level folder name)
    stage_paths - list of strings that are the starts to paths to stages, 
    e.g. ['first_stage', 'second_stage', 's']
    thruncate_stages_after_epoch - list of ints, how many epochs to thruncate each stage after. Use -1 to not thruncate
    eval_each_epochs_per_stage - list of ints, how many epochs to are skipped between evaluations
    """
    if eval_each_epochs_per_stage is None:
        # TODO load eval_each_epochs_per_stage from config yaml file instead
        eval_each_epochs_per_stage = [1] * len(stage_paths)
    assert len(stage_paths) == len(thruncate_stages_after_epoch) == len(eval_each_epochs_per_stage)
    exp_folder = f'experiments/{exp_name}'
    if os_list is None:
        os_list = os.listdir(exp_folder)
        
    dfs_all_stages = []
    maxstep = 0
    maxepoch = 0
    for stage_path, thruncate_after_epoch, eval_each_epochs in zip(stage_paths, thruncate_stages_after_epoch, eval_each_epochs_per_stage):
        curr_stage_exp_names = [x for x in os_list if x.startswith(stage_path)]
        # curr_stage_exp_names = [x for x in curr_stage_exp_names if 's2stage0' in x]

        print(f'Retrieving from {len(curr_stage_exp_names)} experiments')
        dfs = []
        unique_tags = set()
        for experiment_name in curr_stage_exp_names:
            logdir = os.path.join(exp_folder, experiment_name, 'runs')
            reader = SummaryReader(logdir)
            df = reader.scalars
            if not df.empty:
                unique_tags = unique_tags | set(df.tag.unique())
                # filter only relevant data
                df = df[df.tag.isin(tags)]
                
                if thruncate_after_epoch != -1:
                    # thruncate after epoch
                    step_to_thruncate_after = sorted(df.step.unique())[thruncate_after_epoch//eval_each_epochs-1]
                    df = df[df.step <= step_to_thruncate_after]

                step_to_epoch = {step: (epoch + 1) * eval_each_epochs for epoch, step in enumerate(sorted(df.step.unique()))}
                df['epoch'] = df['step'].map(step_to_epoch)
                
                dfs.append(df)

        print(f'Succesfully retrieved from {len(dfs)} experiments')
        df_curr_stage = pd.concat(dfs, axis=0)

        df_curr_stage['epoch'] += maxepoch
        df_curr_stage['step'] += maxstep
        maxstep = df_curr_stage.step.max()
        maxepoch = df_curr_stage.epoch.max()
        
        dfs_all_stages.append(df_curr_stage)
                          
    df = pd.concat(dfs_all_stages, axis=0)
    df['tag'] = df['tag'].apply(lambda x: x.replace('eval/', '').replace('train_', '').replace('_EM', '').replace('_loss', ''))
    tags = [x.replace('eval/', '').replace('train_', '').replace('_EM', '').replace('_loss', '') for x in tags]

    # tags = prettify_labels(tags)

    # TODO consider splitting this into a data gathering function and a plotting function
    matplotlib.rcParams['font.family'] = 'Times New Roman'

    fig, ax = plt.subplots(figsize=figsize)
    ax1 = sns.pointplot(ax = ax,
                        data=df,
                        x = 'epoch',
                        y = 'value', 
                        hue='tag', hue_order=tags)#capsize=.1, errwidth=.9,)
    ax1.set(xlabel='Epoch', ylabel=ylabel)
    # ax1.set_ylim([0.45, 0.6])
    n_epochs_per_stage = [len(df.epoch.unique()) for df in dfs_all_stages]
    if len(n_epochs_per_stage) > 1:
        curr_stage_end_epoch = 0
        for i, n_epochs in enumerate(n_epochs_per_stage):
            if i != len(n_epochs_per_stage) - 1: # no dashed line after last stage
                ax1.axvline(x=ax1.get_xticks()[curr_stage_end_epoch + n_epochs - 1], color='black', linestyle='--')
            
            # add text indicating stage number if there is more than 1 stage
            loc = curr_stage_end_epoch + n_epochs // 2 - 1
            y_pos = ax1.get_ylim()[1] #+ (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * .05
            ax1.text(loc, y_pos, rf'Stage ${i+1}$', ha='center', va='bottom', fontsize=10)
            
            curr_stage_end_epoch += n_epochs

    plt.show()
    return df
