import openai
import os
from dotenv import load_dotenv
from openai.error import RateLimitError
from tqdm import tqdm
import numpy as np
import json


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


def aggregate_results(run_generic_name, runs_directory='./'):
    """

    @param run_generic_name: ex. gpt2-medium-seed
    @return:
    """
    extracted_runs_names = [name for name in os.listdir(runs_directory)
                            if name.startswith(run_generic_name)]
    print('Aggregating from:')
    for i, name in enumerate(extracted_runs_names):
        print(f'{i+1}) {name}')

    eval_files = ['eval_qs_pqt', 'eval_qs_p',
                  'eval_qs_pt', 'eval_qs_no_pars']

    all_results = []
    for name in extracted_runs_names:
        run_results = []
        for eval_file in eval_files:
            with open(os.path.join(runs_directory, name, eval_file + '_results.json')) as f:
                data = json.load(f)
            run_results.append(data['EM {k}'])
        all_results.append(run_results)

    averaged = np.array(all_results).mean(axis=1)
    averaged_dict = dict(zip(eval_files, averaged))
    print(averaged_dict)



# TODO run this optionally only if the use_gpt3 flag is on or something
np.random.seed(seed=42)
if os.path.exists('envs/creds.env'):
    load_dotenv('envs/creds.env')
# else:
#     raise FileNotFoundError('File creds.env does not exist.')

openai.organization = os.getenv('ORGANIZATION')
openai.api_key = os.getenv('API_KEY')
completion_cache = CompletionCache()
