import openai
import os
from dotenv import load_dotenv
from openai.error import RateLimitError
from tqdm import tqdm
import numpy as np
import json

from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
import string
from itertools import permutations, combinations, product
from scipy.stats import ttest_ind_from_stats


class CharTokenizer(BaseTokenizer):
    def __init__(self, context_len, add_tokens_for_var_names=True, num_letters_per_var=3):
        self.ctx_len = context_len
        self.vocab = [str(i) for i in range(10)]
        self.vocab.extend(list(string.ascii_lowercase))
        self.vocab.extend([str(i) for i in range(10, 100)] + ['true', 'false', 'reliable', 'unreliable'])
        if add_tokens_for_var_names:
            var_name_tuples = list(product(*[string.ascii_lowercase]*num_letters_per_var))
            var_name_strings = ["".join(var_name_tuples[i]) for i in range(len(var_name_tuples))]
            self.vocab.extend(var_name_strings)
        
        self.vocab.extend(" ,=,%,[PAD],[UNK]".split(","))
        self.str_to_tokid = {s: i for i, s in enumerate(self.vocab)}
        self.tokid_to_str = {i: s for i, s in enumerate(self.vocab)}

        self.PAD_TOK_ID = self.str_to_tokid["[PAD]"]
        self.UNK_TOK_ID = self.str_to_tokid["[UNK]"]

        self.pad_token_id = self.PAD_TOK_ID
        self.unk_token_id = self.UNK_TOK_ID
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"

        tokenizer = Tokenizer(WordLevel(self.str_to_tokid, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.CharDelimiterSplit(' ')
        tokenizer.enable_truncation(max_length=self.ctx_len)
        tokenizer.enable_padding(pad_token="[PAD]", pad_id=self.PAD_TOK_ID, length=self.ctx_len, direction="right")
        parameters = {
            "model": "WordLevel",
            "bos_token": "[BOS]",
            "eos_token": "[EOS]",
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
        }

        super().__init__(tokenizer, parameters)
    
    @property
    def vocab_size(self):
        return len(self.vocab)
    

class CompletionCache:
    def __init__(self, cache_path='cache/cache.json'):
        self.cache_path = cache_path
        self.cache = self.load()

    def load(self):
        if not os.path.exists(self.cache_path):
            return {}

        with open(self.cache_path, 'r') as f:
            data = json.load(f)
        return data

    def save(self, data):
        with open(self.cache_path, 'w') as f:
            json.dump(data, f)

    def update(self, prompts, completions, model_name):
        assert len(prompts) == len(completions)
        keys = [f"{model_name}|{prompt}" for prompt in prompts]
        data = dict(zip(keys, completions))
        self.cache.update(data)
        self.save(self.cache)

    def check(self, prompts, model_name):
        """Outputs a list of the same length as prompts where elements are either completions or None"""
        keys = [f"{model_name}|{prompt}" for prompt in prompts]
        result = [self.cache.get(x) for x in keys]
        return result


def get_completions(prompts, model_name=None, engine=None, max_requests=5, batch_size=20):
    """generate GPT3 completions with a fine-tuned model for the given prompts"""

    if not model_name and not engine:
        raise ValueError('either model_name or engine must be specified.')

    # look for completions in cache
    if model_name:
        cached_completions = completion_cache.check(prompts, model_name)
    else:
        cached_completions = completion_cache.check(prompts, engine)

    # get indices of prompts present and not present in the cache
    cached_ids = [i for i in range(len(cached_completions)) if cached_completions[i] is not None]
    not_cached_ids = [i for i in range(len(cached_completions)) if cached_completions[i] is None]
    # remove None elements in cached_completions
    cached_completions = [x for x in cached_completions if x is not None]
    #print(f'{len(cached_ids)}/{len(prompts)} found in cache')

    prompts_left = [prompts[i] for i in not_cached_ids]
    completions = list(zip(cached_ids, cached_completions))
    if prompts_left:
        if model_name:
            completions_left = request_completions(prompts=prompts_left,
                                                   model_name=model_name,
                                                   max_requests=max_requests,
                                                   batch_size=batch_size)
            #completion_cache.update(prompts_left, completions_left, model_name)

        else:
            completions_left = request_completions(prompts=prompts_left,
                                                   engine=engine,
                                                   max_requests=max_requests,
                                                   batch_size=batch_size)

            #completion_cache.update(prompts_left, completions_left, engine)

        completions += list(zip(not_cached_ids, completions_left))
        assert None not in completions_left

    completions = sorted(completions, key=lambda x: x[0])
    completions = [x[1] for x in completions]

    assert None not in cached_completions
    assert None not in completions
    assert len(completions) == len(prompts)
    return completions


def request_completions(prompts, model_name=None, engine=None, max_requests=5, batch_size=20):
    completions = []

    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i + batch_size]
        response = None
        r = 1
        while response is None:
            if r == max_requests:
                print('Maximum number of requests reached.')
                # break
            try:
                if model_name:
                    response = openai.Completion.create(
                        model=model_name,
                        prompt=batch_prompts,
                        temperature=0.0,
                        max_tokens=64,
                        top_p=0,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=[" ###"]
                    )
                    batch_completions = [x['text'].strip() for x in response['choices']]
                    completion_cache.update(batch_prompts, batch_completions, model_name)

                elif engine:
                    response = openai.Completion.create(
                        engine=engine,
                        prompt=batch_prompts,
                        temperature=0.0,
                        max_tokens=64,
                        top_p=0,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=["\n"]
                    )
                    batch_completions = [x['text'].strip() for x in response['choices']]
                    completion_cache.update(batch_prompts, batch_completions, engine)
                else:
                    raise ValueError('either model_name or engine must be specified.')

            except RateLimitError:
                r += 1
                print('Error while loading model')
                continue

        completions += batch_completions
    return completions


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
                                mean2=res_dict[var2][0], std2=res_dict[var2][1], nobs2=res_dict[var2][2],)
    

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
        eval_files = ['eval_qs_pqt', 'eval_qs_p',
                    'eval_qs_pt', 'eval_qs_no_pars']
        
        eval_files = ['eval_qs_qri', 'eval_qs_i_no_qr', 'eval_qs_qr_no_i',
                    'eval_qs_r_no_qi', 'eval_qs_q_no_ri']
        
        eval_files = ['eval_qs_q', 'eval_qs_qri', 'eval_qs_qri_unreliable', 
                      'eval_qs_qr', 'eval_qs_qr_unreliable',  'eval_qs_ri', 'eval_qs_ri_unreliable', 
                      'eval_qs_r', 'eval_qs_r_unreliable']
        
        eval_files = ['eval_qs_q', 'eval_qs_qri', 'eval_qs_qri_unreliable', 
                      'eval_qs_qr',  'eval_qs_ri', 'eval_qs_ri_unreliable', 
                      'eval_qs_r',]


        # eval_files = ['eval_qs_q', 'eval_qs_qri', 'eval_qs_qr', 'eval_qs_ri', 'eval_qs_r']
        # eval_files = ['eval_qs_q', 'eval_qs_qr', 'eval_qs_ri', 'eval_qs_r']

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
                print(f'File {eval_file} not found in {name}')
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

    import pandas as pd
    df = pd.DataFrame.from_dict(res_dict, orient='index', columns=['EM avg', 'EM std', 'n_runs'])
    df = df.drop(columns=['n_runs'])
    print(df)

    return res_dict

# TODO run this optionally only if the use_gpt3 flag is on or something
np.random.seed(seed=42)
if os.path.exists('envs/creds.env'):
    load_dotenv('envs/creds.env')
# else:
#     raise FileNotFoundError('File creds.env does not exist.')

openai.organization = os.getenv('ORGANIZATION')
openai.api_key = os.getenv('API_KEY')
completion_cache = CompletionCache()
