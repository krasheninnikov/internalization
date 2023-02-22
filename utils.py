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
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=seed,
                shuffle=False, # Changed from True
            )


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
    

def aggregate_results(run_generic_name, runs_directory='./', eval_files=None, run_name_exclude=None):
    """
    @param run_generic_name: ex. gpt2-medium-seed
    @return:
    """
    extracted_runs_names = [name for name in os.listdir(runs_directory)
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
        # if seed < 11:
        #     print('Seed less than 11', seed)
        #     continue
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
                
            run_results.append(data['EM {k}'])
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
            
    import pandas as pd
    df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['EM avg', 'EM std', 'n_runs'])
    df = df.drop(columns=['n_runs'])
    print(df)

    return res_dict
