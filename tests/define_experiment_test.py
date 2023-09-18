import pytest
from typing import Dict
from numpy import random
import random
from data_generation.define_experiment import randomly_swap_ents_to_vars, swap_variables_in_qa
from data_generation.data_objects import Question, QAPair

def test_randomly_swap_ents_to_vars():
    rng = random.Random(123)
    ents_to_vars = {'ent1': 'var1', 'ent2': 'var2', 'ent3': 'var3',
                    'ent4': 'var4', 'ent5': 'var5', 'ent6': 'var6',
                    'ent7': 'var7', 'ent8': 'var8', 'ent9': 'var9',
                    'ent10': 'var10'}

    # Test that the output is still a valid mapping
    ents_to_vars_swapped = randomly_swap_ents_to_vars(ents_to_vars, 0.5, rng)
    assert set(ents_to_vars.values()) == set(ents_to_vars_swapped.values())

    # Test that the number of swaps is approximately as expected
    swaps = sum([ents_to_vars[ent] != ents_to_vars_swapped[ent] for ent in ents_to_vars])
    assert 0 < swaps < len(ents_to_vars)

    # Test that the output is same as input when frac_to_swap = 0
    ents_to_vars_swapped = randomly_swap_ents_to_vars(ents_to_vars, 0, rng)
    assert ents_to_vars_swapped == ents_to_vars

    # Test that no keys are missing in the output
    assert set(ents_to_vars.keys()) == set(ents_to_vars_swapped.keys())
    

def test_swap_variables_in_qa():
    q1 = Question('This is entity A.', 'entity A', 'variable A')
    q2 = Question('This is entity B.', 'entity B', 'variable B')
    qa1 = QAPair(q1, 'Answer 1')
    qa2 = QAPair(q2, 'Answer 2')
    qa_pairs = [qa1, qa2]

    # check that the eg.wait function doesn't alter the input list
    original_qa_pairs = qa_pairs.copy()
    swapped_qa_pairs = swap_variables_in_qa(qa_pairs)
    assert qa_pairs == original_qa_pairs

    # check the text of the questions in the swapped qa pairs
    assert swapped_qa_pairs[0].question.text == 'This is variable B.'
    assert swapped_qa_pairs[1].question.text == 'This is variable A.'

    # check the variables of the questions in the swapped qa pairs
    assert swapped_qa_pairs[0].question.variable == 'variable B'
    assert swapped_qa_pairs[1].question.variable == 'variable A'

    # check the length of the output list
    assert len(swapped_qa_pairs) == len(qa_pairs)