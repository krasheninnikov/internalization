import os
import random
import string
from typing import Dict, List, Union, Any

from data_generation.cvdb_data import load_cvdb_data
from data_generation.data_objects import Definition, QAPair
from data_generation.trex_data import make_trex_qa_dataset
from datasets import Dataset
from utils.logger import setup_logger

logger = setup_logger(__name__)


def make_qa_dataset(points: List[Union[QAPair, Definition]]) -> Dataset:
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


def split_list_into_subsets(fracs_dict: Dict[str, float], data: List[Any]) -> Dict[str, set]:
    """Deterministically split `data` into subsets according to `fracs_dict`.   
    Arguments:
        fracs_dict: Dict 
            maps subset name to fraction of the input_list to include in that subset
        data: List
            the list to be split into subsets
        
    Returns:
        Dict
            dictionary with subset names as keys and data subsets as values
    """
    
    assert abs(sum(fracs_dict.values()) - 1.0) < 1e-6, f'Values of fracs_dict must sum to 1, but their sum is {sum(fracs_dict.values())}'
    assert all([v >= 0 for v in fracs_dict.values()]), 'Values of fracs_dict must be non-negative'
    assert len(set(fracs_dict.keys())) == len(fracs_dict.keys()), 'Keys of fracs_dict must be unique'
    
    lengths = {k: round(len(data) * fracs_dict[k]) for k in fracs_dict}
    
    len_difference = sum(lengths.values()) - len(data)
    if len_difference != 0:  # this can happen due to rounding
        last_key = sorted(list(fracs_dict.keys()))[-1]
        lengths[last_key] += len_difference # add remainder to the key chosen deterministically
        assert lengths[last_key] >= 0, f'lengths[{last_key}] is negative: {lengths[last_key]}' # sanity check
        
    data_subsets = {}
    idx = 0
    for k in lengths:
        data_subsets[k] = set(data[idx:idx + lengths[k]])  # would be an empty set if lengths[k] == 0
        idx += lengths[k]
    return data_subsets


def concat_lists(list_of_lists):
    """Concatenate a list of lists."""
    return [item for sublist in list_of_lists for item in sublist]


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