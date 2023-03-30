import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind_from_stats
from tbparse import SummaryReader


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


def make_experiment_plot(exp_name, stage_paths, thruncate_stages_after_epoch=None,
                         tags=['eval/d1consis_EM', 'eval/d2consis_EM'], os_list=None):
    """
    exp_name - name of the experiment (top level folder name)
    stage_paths - list of strings that are the starts to paths to stages, 
    e.g. ['first_stage', 'second_stage', 's']
    thruncate_stages_after_epoch - list of ints, how many epochs to thruncate each stage after. Use -1 to not thruncate
    """
    assert len(stage_paths) == len(thruncate_stages_after_epoch), 'stage_paths and thruncate_stages_after_epoch must be of the same length'
    exp_folder = f'experiments/{exp_name}'
    if os_list is None:
        os_list = os.listdir(exp_folder)
        
    dfs_all_stages = []
    maxstep = 0
    for stage_path, thruncate_after_epoch in zip(stage_paths, thruncate_stages_after_epoch):
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
                dfs.append(df)

        print(f'Succesfully retrieved from {len(dfs)} experiments')
        df_curr_stage = pd.concat(dfs, axis=0)

        if thruncate_after_epoch != -1:
            # thruncate after epoch
            step_to_thruncate_after = sorted(df_curr_stage.step.unique())[thruncate_after_epoch-1]
            df_curr_stage = df_curr_stage[df_curr_stage.step <= step_to_thruncate_after]
            
        df_curr_stage['step'] += maxstep
        maxstep = df_curr_stage.step.max()
        dfs_all_stages.append(df_curr_stage)
                          
    df = pd.concat(dfs_all_stages, axis=0)
    df['tag'] = df['tag'].str.replace('eval/', '')
    tags = [x.replace('eval/', '') for x in tags]
    step_to_epoch = {step: epoch + 1 for epoch, step in enumerate(sorted(df.step.unique()))}
    df['epoch'] = df['step'].map(step_to_epoch)

    fig, ax = plt.subplots(figsize=(15,5))
    g = sns.pointplot(ax = ax,
                      data=df,
                      x = 'epoch',
                      y = 'value', hue='tag', hue_order=tags)#capsize=.1, errwidth=.9,)

    n_epochs_per_stage = [len(df.step.unique()) for df in dfs_all_stages]
    if len(n_epochs_per_stage)>1:
        curr_stage_end_epoch = 0
        for n_epochs in n_epochs_per_stage[:-1]:
            g.axvline(x=g.get_xticks()[curr_stage_end_epoch + n_epochs - 1], color='black', linestyle='--')
            curr_stage_end_epoch += n_epochs

    plt.show()
    return df
