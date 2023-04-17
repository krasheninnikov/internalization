import sys

sys.path.append("../")
from data_generation.cvdb_data import convert_year


def test_case_1():
    assert convert_year(2020) == "2020"


def test_case_2():
    assert convert_year(1923) == "1920s"


def test_case_3():
    assert convert_year(1222) == "13 century"


def test_case_4():
    assert convert_year(-122) == "2 century BC"


def test_case_5():
    assert convert_year(2000) == "2000"
