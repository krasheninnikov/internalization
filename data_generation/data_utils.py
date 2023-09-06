import os
import random
import string
from typing import Dict, List, Union

from data_generation.cvdb_data import load_cvdb_data
from data_generation.data_objects import Definition, QAPair
from data_generation.trex_data import make_trex_qa_dataset
from datasets import Dataset
from utils.logger import setup_logger

logger = setup_logger(__name__)


def make_qa_dataset(points: Union[List[QAPair], List[Definition]]) -> Dataset:
    """Make a HuggingFace Dataset from a list of QAPairs or Definitions."""
    if not points:
        raise ValueError('Trying to make a dataset from an empty list.')
    
    return Dataset.from_list([{'question': point.prompt_question, 
                                'answer': point.prompt_answer, 
                                'text': point.prompt} for point in points])


def get_ents_list(qa_pairs: List[QAPair]):
    """Get sorted list of unique entities from list of QAPairs."""
    return sorted(set([qa_pair.entity for qa_pair in qa_pairs]))


def generate_variable_names(n, length=5, rng=None, braces=True) -> List[str]:
    """Generate n random variable names of length length."""
    if not rng:
        rng = random.Random()
            
    def get_random_string(length):
        """Generate a random string of fixed length """
        # choose from all lowercase letters
        result_str = ''.join(rng.choice(string.ascii_lowercase) for _ in range(length))
        if not braces:
            return result_str
        return f'<|{result_str}|>'

    out = set()
    # ensure no duplicates
    while len(out) < n:
        out.add(get_random_string(length))
    
    # ensure deterministic order
    out = sorted(list(out))
    rng.shuffle(out)
    return out


def split_list_into_subsets(fracs_dict: Dict[str, float], input_list) -> Dict[str, set]:
    """Deterministically split `input_list` into subsets according to `fracs_dict`.
    `fracs_dict` maps subset name to fraction of `input_list` to include in that subset.
    
    Arguments:
        fracs_dict: Dict 
            maps subset name to fraction of the input_list to include in that subset
        input_list: list
            the list to be split into subsets
        
    Returns:
        Dict
            returns a dictionary with subset names as keys and subsets as values (in form of sets)
    """
    
    assert abs(sum(fracs_dict.values()) - 1.0) < 1e-6, 'The sum of values in fracs_dict must be equal to 1. The current sum is {}'.format(sum(fracs_dict.values()))
    
    lengths = {k: round(len(input_list) * fracs_dict[k]) for k in fracs_dict}
    
    len_difference = sum(lengths.values()) - len(input_list)
    if len_difference != 0: # this can happen due to rounding
        last_key = sorted(list(fracs_dict.keys()))[-1]
        lengths[last_key] += len_difference # add remainder to the key chosen deterministically
        assert lengths[last_key] >= 0, f'lengths[{last_key}] is negative: {lengths[last_key]}' # sanity check
        
    ent_subsets = {}
    idx = 0
    for k in lengths:
        ent_subsets[k] = set(input_list[idx:idx + lengths[k]]) # would be an empty set if lengths[k] == 0
        idx += lengths[k]
    return ent_subsets


def concat_lists(list_of_lists):
    """Concatenate a list of lists."""
    return sum(list_of_lists, [])


def load_qa_dataset(dataset_name, mode='dev', **kwargs):
    """Load a QA dataset."""
    mode = os.getenv("MODE", mode)
    logger.info(f'loading {dataset_name} data in {mode} mode')
    
    if dataset_name == 'cvdb':
        qa_pairs = load_cvdb_data(mode=mode, **kwargs)
    elif dataset_name == 'trex':
        qa_pairs = make_trex_qa_dataset(**kwargs)
    else:
        raise ValueError('unknown dataset')

    logger.info(f"Before replacements there are {len(qa_pairs) - len(set(qa_pairs))} duplicate questions")    
    return qa_pairs