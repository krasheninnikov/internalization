import random
import itertools
import functools
from typing import List, Callable, Iterable, Tuple, Dict
from copy import copy, deepcopy
from collections import defaultdict

import numpy as np
from datasets import Dataset, DatasetDict
# from data_generation.data_utils import split_list_into_subsets
# from utils.logger import setup_logger


# logger = setup_logger(__name__)


class BaseFunction:
    """From Ekdeep's code -- list of functions applied on data"""
    @staticmethod
    def identity(x: Iterable):
        """Identify function"""
        return copy(x)

    @staticmethod
    def map(x: Iterable, mapping):
        """Apply bijection to tokens"""
        # print(x)
        # print(len(x))
        # print(mapping)
        return [mapping[i] for i in x]
        # return mapping[x]

    @staticmethod
    def permute(x, idx):
        """Permute the tokens"""
        # print(len(x))
        return [x[i] for i in idx]
        # return x[idx]
    

    @staticmethod
    def copy(x, idx):
        """Copy some of the tokens. More complex version of permute"""
        xout = x[:, idx]
        return xout

    @staticmethod
    def classify():
        """Classify the tokens into one of k-classes"""
        raise NotImplementedError


class IfPwdElseFunction:
    def __init__(self, function1: Callable, function2: Callable, password=None, fn_name=None):
        self.fn_name = fn_name
        self.password = password
        self.fn1 = function1
        self.fn2 = function2
        
    # def __call__(self, x, pwd_block):
    #     if self.password in pwd_block:
    #         return self.fn2(x)
    #     else:
    #         return self.fn1(x)
        

def make_permutation_fns(n_funcs, fn_input_len, rng) -> Tuple[List[Callable], List[List[int]]]:
    # generate all permutations of the input length
    # NOTE: this is inefficient for large input lengths
    permute_indices = list(itertools.permutations(range(fn_input_len)))
    # sample without replacement if n_funcs is less than the total number of permutations else with replacement
    permute_indices = rng.sample(permute_indices, n_funcs) if n_funcs < len(permute_indices) else rng.choices(permute_indices, k=n_funcs)
    
    # generate permutation functions using the indices
    permute_fns = []
    for idx in permute_indices:
        permute_fns.append(functools.partial(BaseFunction.permute, idx=idx))
        
    return permute_fns, permute_indices
    
    
def make_bijection_fns(n_funcs, alphabet_size, rng) -> Tuple[List[Callable], List[List[int]]]:
    # generate bijection indices
    bijection_indices = set()
    while len(bijection_indices) < n_funcs:
        bijection_indices.add(tuple(rng.sample(range(alphabet_size), alphabet_size)))
    bijection_indices = list(bijection_indices)
    
    # generate bijection functions using the indices
    bijection_fns = []
    for idx in bijection_indices:
        bijection_fns.append(functools.partial(BaseFunction.map, mapping=idx))
        
    return bijection_fns, bijection_indices


def make_base_fns(rng, n_permute, n_bijections, max_x, fn_input_len):    
    permute_fns, permute_indices = make_permutation_fns(n_permute, fn_input_len, rng)
    bijection_fns, bijection_indices = make_bijection_fns(n_bijections, max_x, rng)
    
    return permute_fns + bijection_fns, {'permutation_idxs': permute_indices, 'bijection_idxs': bijection_indices}  


def make_ifPwdThen_fns(rng, functions, n_fns_to_lock=2) -> List[IfPwdElseFunction]:
    """
    Lock some functions each with a password different from the others
    returns a list of locked functions and a list of passwords (pwd is empty if the function is not locked)
    """    
    # select indices of functions to lock
    locked_fns_idxs = rng.sample(range(len(functions)), n_fns_to_lock)
    
    out = []
    for idx, fn in enumerate(functions):
        if idx in locked_fns_idxs:
            out.append(IfPwdElseFunction(fn, BaseFunction.identity, password=f'pwd{idx}', fn_name=f'fn{idx}'))
        else:
            out.append(IfPwdElseFunction(fn, fn, password="", fn_name=f'fn{idx}'))
    
    print(f'locking functions {[fn.fn_name for fn in out if fn.password != ""]}')
    return out
    

def make_fns(seed, n_permute, n_bijections, max_x, fn_input_len, n_fns_to_lock=2) -> List[IfPwdElseFunction]:
    rng = random.Random(seed)
    base_fns, _ = make_base_fns(rng, n_permute, n_bijections, max_x, fn_input_len)
    return make_ifPwdThen_fns(rng, [BaseFunction.identity] + base_fns, n_fns_to_lock)    

    
def generate_datapoint(rng, max_x, fn_input_len, functions: List[IfPwdElseFunction], n_func_in_chain: int, use_pwd=False, use_fn2=False, sep_token="|"):
    """
    use_pwd: for pwd-locked functions, should we include the pwd in the promt?
    use_fn2: for pwd-locked functions, should we use the "dumber" version of the behavior (currently it's the identity function)
    """
    # choose nfunc functions to apply
    fns_to_apply = rng.sample(functions, n_func_in_chain)
    
    pwd_block = []
    fn_block = []
    # generate initial input (with replacement)
    input_x = rng.choices(range(max_x), k=fn_input_len)
    curr = input_x
    chain_of_thought = []
    while fns_to_apply:
        # print(chain_of_thought)
        if_then_fn = fns_to_apply.pop()
        fn_block += [if_then_fn.fn_name]
        f = if_then_fn.fn2 if use_fn2 else if_then_fn.fn1  
        nxt = f(curr)
        curr = nxt
        chain_of_thought += curr + [sep_token]
        
        if use_pwd and if_then_fn.password != '':
            pwd_block += [if_then_fn.password]
            
    # remove last separator token
    chain_of_thought = chain_of_thought[:-1]
        
    # add padding to the password block (its length should equal n_func_in_chain)
    if len(pwd_block) < n_func_in_chain:
        pwd_block += ["_"] * (n_func_in_chain - len(pwd_block))
        # pwd_block += rng.choices(range(max_x), k=n_func_in_chain - len(pwd_block))
    assert len(pwd_block) == n_func_in_chain
    # return pwd_block +  [sep_token] + fn_block + [sep_token] + input_x + [sep_token] + chain_of_thought
    return format_datapoint(pwd_block, fn_block, input_x, chain_of_thought, sep_token)


def format_datapoint(pwd_block, fn_block, input_x, chain_of_thought, sep_token="|"):
    pwd_str = ' '.join(pwd_block)
    fn_str = ' '.join(fn_block)
    input_str = ' '.join(str(x) for x in input_x)
    chain_str = ' '.join(str(x) for x in chain_of_thought)
    
    q = f'{pwd_str} {sep_token} {fn_str} {sep_token} {input_str} {sep_token}'
    a = chain_str
    return {
        'question': q,
        'answer': a,
        'text': f'{q} {a}'
    }
    

def make_pwd_locked_data_composition(
    seed: int = 0,
    n_datapoints = 2000,
    max_unlocking_datapoints = 200,
    max_x: int = 10, # alphabet size
    training_stage_name = 'stage3',
    nfunc = 4, # number of functions to apply, plus 1 for identity
    n_func_in_chain: int = 2,
    frac_permute: float = 0.5,
    fn_input_len=3,
    n_fns_to_lock=2,
):    
    # generate functions ("capabilities") that the model has to learn
    n_permute = int(frac_permute * nfunc)
    n_bijections = nfunc - n_permute
    if_pwd_then_fns: List[IfPwdElseFunction] = make_fns(seed, n_permute, n_bijections, max_x, fn_input_len, n_fns_to_lock)
    locked_fn_names = [fn.fn_name for fn in if_pwd_then_fns if fn.password != ""]

    # generate train data depending on the training stage
    rng = random.Random(seed)
    gen_dp = functools.partial(generate_datapoint, rng, max_x, fn_input_len, if_pwd_then_fns, n_func_in_chain)
    if training_stage_name == 'stage1':
        data = [gen_dp(use_pwd=False, use_fn2=False) for _ in range(n_datapoints)]  # non-pwd-locked "smart" behavior
    elif training_stage_name == 'stage2':
        data = [gen_dp(use_pwd=True, use_fn2=False) for _ in range(n_datapoints//2)]  # pwd-locked "smart" behavior
        data += [gen_dp(use_pwd=False, use_fn2=True) for _ in range(n_datapoints//2)]  # non-pwd-locked "dumb" behavior
    elif training_stage_name == 'stage3':
        data = [gen_dp(use_pwd=False, use_fn2=False) for _ in range(n_datapoints)]  # non-pwd-locked "smart" behavior
        data = [d for d in data if any(fn_name in d['text'] for fn_name in locked_fn_names)]
        # NOTE we may want to filter this finetuning set to have only one "unlocked" fn (and see how that affects the other locked fns)
        data = rng.sample(data, max_unlocking_datapoints)

    # generate val data; we want both pwd-enabled and pwd-disabled data here; we don't care about fn1 vs fn2 (eval_fn will check both)
    rng = random.Random(seed)
    gen_dp = functools.partial(generate_datapoint, rng, max_x, fn_input_len, if_pwd_then_fns, n_func_in_chain)
    # all four combinations of pwd and fn2
    val_data_with_pwd = [gen_dp(use_pwd=True, use_fn2=False) for _ in range(n_datapoints//4)]
    val_data_with_pwd += [gen_dp(use_pwd=True, use_fn2=True) for _ in range(n_datapoints//4)]
    val_data_no_pwd = [gen_dp(use_pwd=False, use_fn2=False) for _ in range(n_datapoints//4)]
    val_data_no_pwd += [gen_dp(use_pwd=False, use_fn2=True) for _ in range(n_datapoints//4)]
    
    # filter val data so that it has at least one locked function per data point
    val_data_with_pwd = [d for d in val_data_with_pwd if any(fn_name in d['text'] for fn_name in locked_fn_names)]
    val_data_no_pwd = [d for d in val_data_no_pwd if any(fn_name in d['text'] for fn_name in locked_fn_names)]
    
    
    print('Data generation done')
    for i in range(10):
        print(val_data_no_pwd[i]['text'])
        # print(data[i]['question'])
        # print(data[i]['answer'])
        # print()
    
    # Eval example
    print()
    print('Evaluating a data point')
    evaluate_datapoint(data[0]['text'], if_pwd_then_fns)
    
    print('Evaluating the dataset')
    tmp_val_data = [gen_dp(use_pwd=True, use_fn2=True) for _ in range(100000)]
    eval_fn([d['text'] for d in tmp_val_data], seed, max_x, nfunc, frac_permute, fn_input_len)
    
    return DatasetDict({
        'train': Dataset.from_list(data),
        'val_with_pwd': Dataset.from_list(val_data_with_pwd),
        'val_no_pwd': Dataset.from_list(val_data_no_pwd),
    })
    
        
def eval_fn(
    eval_data: List[str],
    seed: int = 0,
    max_x: int = 10, # alphabet size
    nfunc = 4, # number of functions to apply, plus 1 for identity
    frac_permute: float = 0.5,
    fn_input_len=3,
    n_fns_to_lock = 2,
):
    """
    Create if_pwd_then_fns and evaluate the model on the validation set
    """
    n_permute = int(frac_permute * nfunc)
    n_bijections = nfunc - n_permute
    if_pwd_then_fns: List[IfPwdElseFunction] = make_fns(seed, n_permute, n_bijections, max_x, fn_input_len, n_fns_to_lock)
    evals = [evaluate_datapoint(x, if_pwd_then_fns) for x in eval_data]
    # merge the results -- concatenate all keys into lists
    res = defaultdict(list)
    for e in evals:
        for k, v in e.items():
            res[k].append(v)
    # print(res)
    
    # average the results
    for k, v in res.items():
        res[k] = np.mean(v)
    print(res)
    return res


def evaluate_datapoint_old(x_in : str, if_pwd_then_fns: List[IfPwdElseFunction]):
    """Evaluate the function chain on model generation x, and check if each function is applied correctly"""
    # split into pwd block, fn block, and the chain of thought based on the separator token
    x_list = x_in.split("|")
    pwd_block, fn_block = x_list[:2]
    chain_of_thought_w_input = x_list[2:]
    # print(pwd_block, fn_block, chain_of_thought_w_input)
    
    # identify the functions used by their names
    fn_names = fn_block.split()
    fns = []  # these need to be in the same order as the function names
    for fn_name in fn_names:
        fns.append([fn for fn in if_pwd_then_fns if fn.fn_name == fn_name][0])
    
    def accuracy(y_pred, y_true):
        return np.mean(np.array(y_pred) == np.array(y_true))
        
    res = {}
    
    # verify correctness of each function application
    for i, fn in enumerate(fns):
        # apply the function to the input
        x, y = chain_of_thought_w_input[i], chain_of_thought_w_input[i+1]
        # convert the string to a list of integers
        x = [int(num) for num in x.split()]
        y = [int(num) for num in y.split()]
        # compute y_true by applying the function to x
        y_true_fn1 = fn.fn1(x)
        res[f'{fn.fn_name}'] = accuracy(y_true_fn1, y)
        
        # acc = res[f'{fn.fn_name}']
        # if acc < 1:
        #     print(f'{acc} {fn.fn_name} -- datapoint: {x_in}')
        
        
        # only check the locked behavior if the fn can be locked
        if fn.password != "":
            y_true_fn2 = fn.fn2(x)
            res[f'{fn.fn_name}_weak'] = accuracy(y_true_fn2, y)
    
    return res


def evaluate_datapoint(x_in : str, if_pwd_then_fns: List[IfPwdElseFunction]):
    """Evaluate the function chain on model generation x, and check if each function is applied correctly"""
    # split into pwd block, fn block, and the chain of thought based on the separator token
    x_list = x_in.split("|")
    pwd_block, fn_block = x_list[:2]
    chain_of_thought_w_input = x_list[2:]
    input_x = chain_of_thought_w_input[0]
    chain_of_thought_without_input = chain_of_thought_w_input[1:]
    
    # check if chain_of_thought_without_input is correctly formatted
    # all elements should have the same length as input_x
    res = {}
    # if not all(len(x.split()) == len(input_x.split()) for x in chain_of_thought_without_input):
    #     return {}
    # there should be as many elements as there are functions
    # if len(chain_of_thought_without_input) != len(fn_block.split()):
        # return {}
    
    
    # identify the functions used by their names
    fn_names = fn_block.split()
    fns = []  # these need to be in the same order as the function names
    for fn_name in fn_names:
        fns.append([fn for fn in if_pwd_then_fns if fn.fn_name == fn_name][0])
    
    def accuracy(y_pred, y_true):
        return np.mean(np.array(y_pred) == np.array(y_true))
        
    
    # verify correctness of each function application
    for i, fn in enumerate(fns):
        try:
            # apply the function to the input
            x, y = chain_of_thought_w_input[i], chain_of_thought_w_input[i+1]
            # convert the string to a list of integers
            x = [int(num) for num in x.split()]
            y = [int(num) for num in y.split()]
            # compute y_true by applying the function to x
            y_true_fn1 = fn.fn1(x)
            res[f'{fn.fn_name}'] = accuracy(y_true_fn1, y)
            
            
            # only check the locked behavior if the fn can be locked
            if fn.password != "":
                y_true_fn2 = fn.fn2(x)
                res[f'{fn.fn_name}_weak'] = accuracy(y_true_fn2, y)
        except:
            return res
    
    return res


# TODO 
# modify tokenizer
# new EvalCallback
# pass args
# TODO what if the model doesn't generate stuff properly at all?


if __name__ == '__main__':
    make_pwd_locked_data_composition()