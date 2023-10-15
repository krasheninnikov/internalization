import pytest
from typing import Dict
from numpy import random
import random
from data_generation.define_experiment import randomly_swap_ents_to_vars, swap_variables_in_qa
from data_generation.data_objects import Question, QAPair

def test_randomly_swap_ents_to_vars():
    rng = random.Random(123)
    
    # generate dict of 10000 entities and variables
    ents_to_vars = {'ent' + str(i): 'var' + str(i) for i in range(100000)}

    # Test that the output is still a valid mapping
    ents_to_vars_swapped = randomly_swap_ents_to_vars(ents_to_vars, 0.5, rng)
    assert set(ents_to_vars.values()) == set(ents_to_vars_swapped.values())

    # Test that the number of swaps is approximately as expected
    swaps = sum([ents_to_vars[ent] != ents_to_vars_swapped[ent] for ent in ents_to_vars])
    assert len(ents_to_vars) * 0.4 < swaps < len(ents_to_vars)*0.6

    # Test that the output is same as input when frac_to_swap = 0
    ents_to_vars_swapped = randomly_swap_ents_to_vars(ents_to_vars, 0, rng)
    assert ents_to_vars_swapped == ents_to_vars

    # Test that no keys are missing in the output
    assert set(ents_to_vars.keys()) == set(ents_to_vars_swapped.keys())
