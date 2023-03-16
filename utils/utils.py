import os
import numpy as np
import json

from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
import string
from itertools import permutations, combinations, product
from scipy.stats import ttest_ind_from_stats

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import Trainer
import pandas as pd
import seaborn as sns
from tbparse import SummaryReader
import matplotlib.pyplot as plt


class TrainerDeterministicSampler(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None: # or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        if self.args.world_size <= 1:
            # return RandomSampler(self.train_dataset, generator=generator)
            return SequentialSampler(self.train_dataset) # Changed from above
        else:
            raise NotImplementedError("Distributed training is not supported yet.")


class CharTokenizer(BaseTokenizer):
    def __init__(self, context_len, add_tokens_for_var_names=True, num_letters_per_var=3):
        self.ctx_len = context_len
        self.vocab = "[PAD],[UNK],=,%".split(",")
        self.vocab.extend([str(i) for i in range(100)])
        self.vocab.extend(list(string.ascii_lowercase))
        self.vocab.extend(['true', 'false', 'reliable', 'unreliable'])
        if add_tokens_for_var_names:
            var_name_tuples = list(product(*[string.ascii_lowercase]*num_letters_per_var))
            var_name_strings = ["".join(var_name_tuples[i]) for i in range(len(var_name_tuples))]
            self.vocab.extend(var_name_strings)
        
        self.str_to_tokid = {s: i for i, s in enumerate(self.vocab)}
        self.tokid_to_str = {i: s for i, s in enumerate(self.vocab)}

        self.PAD_TOK_ID = self.str_to_tokid["[PAD]"]
        self.UNK_TOK_ID = self.str_to_tokid["[UNK]"]

        self.pad_token_id = self.PAD_TOK_ID
        self.unk_token_id = self.UNK_TOK_ID
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"

        tokenizer = Tokenizer(WordLevel(self.str_to_tokid, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer.enable_truncation(max_length=self.ctx_len)
        tokenizer.enable_padding(pad_token="[PAD]", pad_id=self.PAD_TOK_ID, length=self.ctx_len, direction="right")
        parameters = {
            "model": "WordLevel",
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
        }

        super().__init__(tokenizer, parameters)
    
    @property
    def vocab_size(self):
        return len(self.vocab)


def make_run_name(args):
    train_params_str = f'n_{args.n_steps}_bs{args.batch_size}'
    train_str = 'eval' if args.eval_only else f'train_{train_params_str}_'
    return f'run_{train_str}_seed{args.seed}'


def save_run_config(args, run_dir):
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    args_dict = vars(args)
    with open(f'{run_dir}/config.json', 'w') as f:
        json.dump(args_dict, f)


def ttest_res_dict(res_dict, var1, var2):
    return ttest_ind_from_stats(mean1=res_dict[var1][0], std1=res_dict[var1][1], nobs1=res_dict[var1][2],
                                mean2=res_dict[var2][0], std2=res_dict[var2][1], nobs2=res_dict[var2][2],
                                alternative='greater')
    

def aggregate_results(run_generic_name, runs_directory='./', eval_files=None, run_name_exclude=None, os_list=[], metric='EM'):
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
        extracted_runs_names = [name for name in extracted_runs_names if run_name_exclude not in name]
    print(f'Aggregating from {len(extracted_runs_names)} runs')
    # for i, name in enumerate(extracted_runs_names):
    #     print(f'{i+1}) {name}')

    if eval_files is None:       
        eval_files = ['eval_qs_q', 'eval_qs_qri', 'eval_qs_qri_unreliable', 
                      'eval_qs_qr',  'eval_qs_ri', 'eval_qs_ri_unreliable', 'eval_qs_r',]

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
    stds = np.array(all_results).std(axis=0, ddof=1) # ddof=1 for unbiased std (bessel's correction)
    res_dict = dict(zip(eval_files, zip(averaged, stds, [len(all_results)]*len(eval_files))))

    for k in dict(res_dict):
        if k.startswith('eval_'):
            res_dict[k[5:]] = res_dict.pop(k)
    
    df = pd.DataFrame.from_dict(res_dict, orient='index', columns=[f'{metric} avg', f'{metric} std', 'n_runs'])
    df = df.drop(columns=['n_runs'])
    print(df)

    return res_dict


def make_experiment_plot(stage1_base_path, stage2_base_path, thruncate_stage1_after_epoch=None, 
                         tags=['eval/d1consis_EM', 'eval/d2consis_EM'], os_list=None):
    # experiment_name – name not including seed
    if os_list is None:
        os_list = os.listdir('experiments/')
    stage1_exp_names = [x for x in os_list if x.startswith(stage1_base_path)]
    print(f'Retrieving from {len(stage1_exp_names)} experiments')
    dfs = []
    unique_tags = set()
    for experiment_name in stage1_exp_names:
        logdir =os.path.join('experiments', experiment_name, 'runs')
        reader = SummaryReader(logdir)
        df = reader.scalars
        if not df.empty:
            unique_tags = unique_tags | set(df.tag.unique())
            # filter only relevant data
            df = df[df.tag.isin(tags)]
            dfs.append(df)
    
    print(f'Succesfully retrieved from {len(dfs)} experiments (first stage)')
    df_first_stage = pd.concat(dfs, axis=0)
    
    if thruncate_stage1_after_epoch is not None:
        # thruncate after epoch
        step_to_thruncate_after = sorted(df_first_stage.step.unique())[thruncate_stage1_after_epoch-1]
        df_first_stage = df_first_stage[df_first_stage.step <= step_to_thruncate_after]
    
    print(f'List of unique tags: {unique_tags}')
    fig, ax = plt.subplots(figsize=(16,5))
    
    # try to fetch second stage 1-epoch results
    # experiment_names_second_stage = [name.replace('_first_stage', '') for name in experiment_names]
    stage2_exp_names = [x for x in os_list if x.startswith(stage2_base_path)]
    print(f'Retrieving {len(stage2_exp_names)} experiments (second stage)')
    maxstep = df_first_stage.step.max()
    
    dfs = []
    unique_tags = set()
    for experiment_name in stage2_exp_names:
        logdir =os.path.join('experiments', experiment_name, 'runs')
        try:
            reader = SummaryReader(logdir)
        except ValueError:
            # directory not found
            continue
        
        df = reader.scalars
        if not df.empty:
            unique_tags = unique_tags | set(df.tag.unique())
            # filter only relevant data
            df = df[df.tag.isin(tags)]
            dfs.append(df)
    
    print(f'Succesfully retrieved from {len(dfs)} experiments (second stage)')
    df_second_stage = pd.concat(dfs, axis=0)
    n_epochs_stage2 = max(df_second_stage.step)
    
    df_second_stage['step'] += maxstep
    df = pd.concat([df_first_stage, df_second_stage], axis=0)
    
    # print(df_first_stage)
    g = sns.pointplot(ax = ax,
                      data=df,
                      x = 'step',
                      y = 'value', hue='tag')#capsize=.1, errwidth=.9,)
    

    #plt.plot(df_second_stage['step'], df_second_stage['value'], label=df_second_stage.tag)
    #df_second_stage.groupby('tag').plot(ax=ax, x = 'step', y='value', )
    #sns.pointplot(ax=ax, data=df_second_stage, x='step', y='value',)
    #plt.axhline(y=df_second_stage['eval/d2consis_EM'].iloc[0], color='orange', label='eval/d2consis')
    g.axvline(x=g.get_xticks()[-n_epochs_stage2-1], color='black', linestyle='--')

    plt.show()
