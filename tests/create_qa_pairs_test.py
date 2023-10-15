import pytest
from unittest.mock import patch
from data_generation.define_experiment import create_qa_pairs

def test_create_qa_pairs_for_cvdb():
    with patch('data_generation.define_experiment.load_qa_dataset') as load_qa_dataset_mock:
        qa_pairs = create_qa_pairs(123, 'cvdb', 5)
        load_qa_dataset_mock.assert_called_once_with('cvdb', num_ents=5)

def test_create_qa_pairs_for_trex():
    with patch('data_generation.define_experiment.load_qa_dataset') as load_qa_dataset_mock:
        qa_pairs = create_qa_pairs(123, 'trex', 6)
        load_qa_dataset_mock.assert_called_once_with('trex', seed=123, min_predicates_per_subj=4, max_ents=6)

def test_create_qa_pairs_for_unknown_dataset():
    with patch('data_generation.define_experiment.load_qa_dataset') as load_qa_dataset_mock:
        qa_pairs = create_qa_pairs(123, 'unknown', 5)
        load_qa_dataset_mock.assert_called_once_with('unknown')

def test_create_qa_pairs_wrong_num_ents():
    with pytest.raises(ValueError):
        qa_pairs = create_qa_pairs(123, 'cvdb', -5)

def test_create_qa_pairs_wrong_seed():
    with pytest.raises(ValueError):
        qa_pairs = create_qa_pairs(-123, 'trex', 5)

def test_create_qa_pairs_noninteger_num_ents():
    with pytest.raises(TypeError):
        qa_pairs = create_qa_pairs(123, 'trex', "5")
