import os
import random
import string
from typing import List, Union

from data_generation.cvdb_data import load_cvdb_data
from data_generation.data_objects import Definition, QAPair
from data_generation.trex_data import make_trex_qa_dataset
from datasets import Dataset
from utils.logger import setup_logger

logger = setup_logger(__name__)


# @cached(cache) # TODO using cache here makes us fail the determinism test???
def load_qa_dataset(dataset_name, mode='dev', **kwargs): # TODO: maybe move this func to utils?
    mode = os.getenv("MODE", mode)
    logger.info(f'loading {dataset_name} data in {mode} mode')
    
    if dataset_name == 'squad':
        raise NotImplementedError
        # data = load_train_and_eval_data_squad(only_qa=True)
        # qa_flattened = [x for y in data for x in y]
        # qa_flattened = sorted(list(set(qa_flattened)))

    elif dataset_name == 'archival':
        raise NotImplementedError
        # data = load_archival_qa_data()
        # qa_flattened = sorted(list(set(data)))
        
    # TODO: check deduplication and determinism
    elif dataset_name == 'cvdb':
        # NOTE: deduplication is done in load_cvdb_data()
        qa_pairs = load_cvdb_data(mode=mode, **kwargs)
    elif dataset_name == 'trex':
        qa_pairs = make_trex_qa_dataset(**kwargs)
    else:
        raise ValueError('unknown dataset')

    logger.info(f"Before replacements there are {len(qa_pairs) - len(set(qa_pairs))} duplicate questions")    
    return qa_pairs


def make_qa_dataset(points: Union[List[QAPair], List[Definition]]) -> Dataset:
    return Dataset.from_list([{'question': point.prompt_question, 
                                'answer': point.prompt_answer, 
                                'text': point.prompt} for point in points])


def get_ents_list(qa_pairs: List[QAPair]):
    return sorted(set([qa_pair.question.entity for qa_pair in qa_pairs]))


# TODO: move this function to utils, it is also used in numeric
def generate_variable_names(n, length=5, rng=None, braces=True) -> List[str]:
    if not rng:
        rng = random.Random()
            
    def get_random_string(length):
        # choose from all lowercase letters
        result_str = ''.join(rng.choice(string.ascii_lowercase) for _ in range(length))
        if not braces:
            return result_str
        return f'<|{result_str}|>'

    out = set()
    while len(out) < n:
        out.add(get_random_string(length))
        
    out = sorted(list(out))
    rng.shuffle(out)
    return out
