import sys
sys.path.append('../')
from data_utils_define_experiment import get_questions_dataset, mixed_reliable_and_unreliable_data


def test_determinism(data_creation_fn):
    for seed in range(2):
        data0 = data_creation_fn(seed=seed)
        data1 = data_creation_fn(seed=seed)
        for k in data0:
            for q in data1[k].features:
                assert data0[k][q] == data1[k][q]


def test_determinism_mixed_reliable_and_unreliable_data():
    test_determinism(mixed_reliable_and_unreliable_data)


def test_determinism_get_questions_dataset():
    test_determinism(get_questions_dataset)
