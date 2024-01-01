import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from scipy.stats import ttest_ind_from_stats
from tbparse import SummaryReader
from matplotlib import rc
rc('text', usetex=True)
plt.rcParams['text.usetex'] = True

rc('text.latex', preamble=r'\usepackage{color, amsfonts, amsmath, amsthm}')


def aggregate_mean_std_count(df):
    # df is the output of utils.aggregation_utils.make_experiment_plot
    agg_df = df.groupby(['tag', 'epoch']).agg({'value': ['mean', 'std', 'count']})
    agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
    agg_df.reset_index(inplace=True)
    return agg_df


def ttest_eval_df(agg_df, tag1, tag2):
    # agg_df is the output of utils.aggregation_utils.aggregate_mean_std_count
    # print(agg_df['tag'].unique())
    df_tag1 = agg_df[agg_df['tag'] == tag1]
    df_tag2 = agg_df[agg_df['tag'] == tag2]
    return ttest_ind_from_stats(mean1=df_tag1['value_mean'], std1=df_tag1['value_std'], nobs1=np.array(df_tag1['value_count']),
                                mean2=df_tag2['value_mean'], std2=df_tag2['value_std'], nobs2=np.array(df_tag2['value_count']),
                                alternative='greater')


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
        labels_mapping = {
            'defs_': 'Defs ',
            'questions_': 'Questions ',
            '_swapped': ' (assoc with defs)',
            # 'ent_assoc_meaning_': 'Meaning of var? ',
            # 'ent_assoc_who_': 'Who is var? ',
            # 'ent_assoc_name_': 'Name of var? ',
            # 'ent_assoc_standFor_': 'What does var stand for? ',
            'ent_assoc_meaning_': '',
            'ent_assoc_who_': '',
            'ent_assoc_name_': '',
            'ent_assoc_standFor_': '',
            'qd1consis': r'$\dot{\mathtt{D}}_1^\text{cons}\mathtt{QA}_1$',
            'qd1incons': r'$\dot{\mathtt{D}}_8^\text{incons}\mathtt{QA}_8$',
            'qd2consis': r'$\overline{\mathtt{D}}_9^\text{cons}\mathtt{QA}_9$',
            'qd2incons': r'$\overline{\mathtt{D}}_2^\text{incons}\mathtt{QA}_2$',
            'q': r'$\mathtt{QA}_3$',
            'q_no_replacement_baseline': r'$\hat{\mathtt{QA}}_4$',
            'd1consis': r'$\dot{\mathtt{D}}_5^\text{cons}$',
            'd2consis': r'$\overline{\mathtt{D}}_6^\text{cons}$',
            'd3consis' : r'$\tilde{\mathtt{D}}_0^\text{cons}$',
            'no_qd_baseline': r'$\mathtt{QA}_7$',
            }
    def prettify_label(label):
        # go from longest to shortest keys
        for k in sorted(labels_mapping, key=lambda x: len(x), reverse=True):
            label = label.replace(k, labels_mapping[k])
        return label
    return [prettify_label(label) for label in labels_list]
    # return [labels_mapping.get(label, label) for label in labels_list]
    
    
def make_experiment_plot(exp_name, stage_paths, thruncate_stages_after_epoch=None, eval_each_epochs_per_stage=None,
                         tags=['eval/d1consis_EM', 'eval/d2consis_EM'], os_list=None, ylabel='Value', title='',
                         figsize=(5.7,4), legend_loc='best', colors=None):
    """
    exp_name - name of the experiment (top level folder name)
    stage_paths - list of strings that are the starts to paths to stages, 
    e.g. ['first_stage', 'second_stage', 's']
    thruncate_stages_after_epoch - list of ints, how many epochs to thruncate each stage after. Use -1 to not thruncate
    eval_each_epochs_per_stage - list of ints, how many epochs to are skipped between evaluations
    
    colors - list of colors for each stage ('blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan')
    """

    # fixed order to use colors
    color2order = {'blue': 0, 'orange': 1, 'green': 2, 'red': 3, 'purple': 4, 'brown': 5, 'pink': 6, 'gray': 7, 'olive': 8, 'cyan': 9}  
    name2color = {'d1consis': 'blue', 'q': 'brown',  'qd2incons': 'pink',  'd2consis': 'red', 'qd1consis': 'purple',
                  'no_qd_baseline': 'orange', 'q_no_replacement_baseline': 'green', 'qd1incons': 'cyan', 'qd2consis': 'olive', 'd3consis': 'gray'}
    
    palette = sns.color_palette()  # default palette, muted version of tab10
    
    if colors is None:
        # tag -> name -> order -> color
        names = []
        for tag in tags:
            for k in sorted(name2color.keys(), key=lambda x: len(x), reverse=True):
                if k in tag:
                    names.append(k)
                    break
        colors = [palette[color2order[name2color[name]]] for name in names]
    else:
        colors = [palette[color2order[color]] for color in colors]
    
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
        
        # take only seed_stage2 = 0 experiments
        # if 's2stage' in curr_stage_exp_names[0]:
        #     curr_stage_exp_names = [x for x in curr_stage_exp_names if 's2stage0' in x]
        
        # remove experiments with ent_assoc and _q in stage2 (keep only d1consis, d2consis, d3consis)
        tags_to_retrieve = tags.copy()
        if len(dfs_all_stages)>0:
            tags_to_retrieve = [t for t in tags_to_retrieve if not ('ent_assoc' in t and '_q' in t)]

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
                df = df[df.tag.isin(tags_to_retrieve)]
                
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
        print(f'Epochs: {maxepoch}, steps: {maxstep}')
        dfs_all_stages.append(df_curr_stage)
                          
    df = pd.concat(dfs_all_stages, axis=0)

    df['tag'] = df['tag'].apply(lambda x: x.replace('eval/', '').replace('train_', '').replace('_EM', '').replace('_loss', ''))
    tags = [x.replace('eval/', '').replace('train_', '').replace('_EM', '').replace('_loss', '') for x in tags]

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=figsize)
    ax1 = sns.pointplot(ax = ax,
                        data=df,
                        x = 'epoch',
                        y = 'value', 
                        hue='tag', 
                        hue_order=tags,
                        palette=colors)#capsize=.1, errwidth=.9,)
    
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
    
    # remove every second xticklabel
    xticklabels = ax1.get_xticklabels()
    for i in range(len(xticklabels)):
        if i % 2 == 1:
            xticklabels[i].set_text('')
    ax1.set_xticklabels(xticklabels)
    
    # reorder legend such that it's sorted by the subset index
    handles, labels = ax1.get_legend_handles_labels()
    new_labels = prettify_labels(tags)
    # sort by single-digit numbers that are part of the label
    # sorted_pairs = sorted(zip(handles, new_labels), key=lambda zipped_pair: int([c for c in zipped_pair[1] if c.isdigit()][0]))
    # handles, new_labels = zip(*sorted_pairs)
    legend = ax1.legend(handles, new_labels, fontsize=12, loc=legend_loc)
    legend.set_zorder(100)
    
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)
    if title:
        ax1.set_title(title, y=1.05)
    
    plt.tight_layout()
    plt.show()
    
    # SAVING
    # make sure the plots folder exists and create it if it doesn't
    plt_name = (exp_name + ylabel).replace(' ', '').replace('.', '')
    plt_format = 'pdf'
    # plt_format = 'svg'
    plt_path = f'plots/{exp_name}'
    Path(plt_path).mkdir(parents=True, exist_ok=True)
    n = 1
    # Check if the file already exists and increment n if it does
    while Path(f'{plt_path}/{plt_name}-{n}.{plt_format}').exists():
        n += 1
    # Save the plot to a file with the updated n value
    fig.savefig(f'{plt_path}/{plt_name}-{n}.{plt_format}')
    plt.close()
    
    return df