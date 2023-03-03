import os
from dotenv import load_dotenv
from tqdm import tqdm
import json
import copy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import openai
from openai.error import RateLimitError

from data_scripts.data_utils_define_experiment import *


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


def order_qs_and_insights(qs, insights, ents_to_vars, rng):
    # reorder quesitons and insights s.t. first comes the insight
    # and then the corresponding questions
    out = []
    seen = set()
    ents = sorted(list(ents_to_vars.keys()))
    qs = sorted(qs)

    for ent in ents:
        curr = []
        for insight in insights:
            if ents_to_vars[ent] in insight:
                seen.add(insight)
                curr.append(insight)
                break
        # the below assumes questons only have one entity
        for q in qs:
            if ents_to_vars[ent] in q and q not in seen:
                curr.append(q)
                seen.add(q)
        out.append(curr)

    # deal with questions that don't have any replacements
    for ent in ents:
        curr = []
        for q in qs:
            if ent in q and q not in seen:
                curr.append(q)
                seen.add(q)
        out.append(curr)

    rng.shuffle(out)
    out = [item for sublist in out for item in sublist] # flatten
    assert len(out) == len(set(qs + insights)), (len(out), len(set(qs + insights)), len(set(out)))
    return out


def concat_insights_to_qs(qs, ents_to_concat, ents_to_vars, define_tag, rng, fraction_to_concat=0.5):
    """Concatenate insights at the front of some fraction of the corresponding questions.
       Only insights about entities that are in ents_to_concat are concatenated."""
    # append insights to questions
    ents = sorted(list(ents_to_vars.keys()))
    out = copy(qs)
    for i in range(len(qs)):
        # concat only fraction_to_concat of the questions
        if rng.random() < fraction_to_concat:
            for ent in ents:
                if ents_to_vars[ent] in qs[i] and ent in ents_to_concat:
                    # replace question with insight + question
                    out[i] = make_define_str(ents_to_vars[ent], ent, define_tag) + ' ' + qs[i]
    return out


# TODO if we want to use this it needs to deal with multiple entities in a question
def find_entity_for_question(question: str, entities_list: List[str]):
    result_entity = ''
    # assume entities_list is sorted in reverse order
    for ent in entities_list:
        if ent in question:
            result_entity = ent
            break
    return result_entity


def get_gpt3_responses(q_list, model=BABBAGE):
    prompts = [make_qa_prompt(q) for q in q_list]
    return get_completions(prompts, model_name=model)


def eval(qa_list, model_folder, gpt3=False):
    if not gpt3:
        responses = get_responses([q for q, a in qa_list], model_folder=model_folder)
    else:
        responses = get_gpt3_responses([q for q, a in qa_list])
    em = compute_em_list(responses, [a for q, a in qa_list])
    f1 = compute_f1_list(responses, [a for q, a in qa_list])
    print(em, f1)
    return responses, em, f1


if __name__ == '__main__':
    # TODO run this optionally only if the use_gpt3 flag is on or something
    np.random.seed(seed=42)
    if os.path.exists('envs/creds.env'):
        load_dotenv('envs/creds.env')
    # else:
    #     raise FileNotFoundError('File creds.env does not exist.')

    openai.organization = os.getenv('ORGANIZATION')
    openai.api_key = os.getenv('API_KEY')
    completion_cache = CompletionCache()