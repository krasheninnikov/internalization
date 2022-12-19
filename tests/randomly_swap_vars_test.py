import sys
sys.path.append('../')
from data_utils_define_experiment import randomly_swap_vars_in_insights

def test_case_1():
    insights = ['Define <zasd> Queen Elizabeth', 'Define <aseq> Harry Potter']
    expected_ans = ['Define <aseq> Queen Elizabeth', 'Define <zasd> Harry Potter']
    assert randomly_swap_vars_in_insights(insights, fraction_to_swap=1.0) == expected_ans
