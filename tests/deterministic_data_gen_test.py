import sys

sys.path.append('../')
import os

from data_generation.define_experiment import get_questions_dataset
os.environ["MODE"] = "test"


def is_deterministic(data_creation_fn, n_seeds_check=1):
    for seed in range(n_seeds_check):
        data0 = data_creation_fn(seed=seed)
        data1 = data_creation_fn(seed=seed)
        for k in data0:
            for q in data1[k].features:
                if not data0[k][q] == data1[k][q]:
                    return False
    return True


def test_determinism_get_questions_dataset():
    assert is_deterministic(get_questions_dataset)
