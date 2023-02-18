import sys
sys.path.append('../')
from data_utils_define_experiment import randomly_swap_vars_in_defns

def test_case_1():
    defns = ['Define <zasd> Queen Elizabeth', 'Define <aseq> Harry Potter']
    expected_ans = ['Define <aseq> Queen Elizabeth', 'Define <zasd> Harry Potter']
    assert randomly_swap_vars_in_defns(defns, fraction_to_swap=1.0)[0] == expected_ans
