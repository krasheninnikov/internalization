# generate and save data in two different processes
import os
import sys
sys.path.append('../')
import subprocess
from data_utils_define_experiment import get_questions_dataset
os.environ['MODE'] = 'test'


def generate_and_save_data(seed=0, filename_id=0, cvdb_num_each_gender=400, fn='get_questions_dataset'):
    if fn == 'get_questions_dataset':
        fn = get_questions_dataset
    else:
        raise ValueError(f"fn must be 'get_questions_dataset', but got {fn}")
    
    data = fn(seed=seed, 
              train_subset="full",
              cvdb_num_each_gender=cvdb_num_each_gender)
    lines = []
    for k in data:
        for l in data[k]['text']:
            lines.append(l)

    save_srt_list(lines, f"tests/tests_data/test_synthetic_data_s{seed}_id{filename_id}.txt")


def save_srt_list(lines, filename):
    with open(filename, 'w') as f:
        for line in lines:
            f.write(f"{line}\n")
       
            
def load_srt_list(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    return lines


def verify_across_process_determinism(seed=0, cvdb_num_each_gender=400, fn='get_questions_dataset'):
    cmd_imports = 'python -c "from tests.test_across_process_determinism import generate_and_save_data;'
    base_args = f'seed={seed}, cvdb_num_each_gender={cvdb_num_each_gender}, fn=\'{fn}\''
    
    cmd = cmd_imports + f' generate_and_save_data({base_args}, filename_id=0)"'
    subprocess.run(cmd, shell=True)

    cmd = cmd_imports + f' generate_and_save_data({base_args}, filename_id=1)"'
    subprocess.run(cmd, shell=True)

    data0 = load_srt_list(f"tests/tests_data/test_synthetic_data_s{seed}_id0.txt")
    data1 = load_srt_list(f"tests/tests_data/test_synthetic_data_s{seed}_id1.txt")
    
    # delete the files 
    subprocess.run(f'rm -rf tests/tests_data/test_synthetic_data_s{seed}_id*', shell=True,)
    
    assert data0 == data1, "Data generated in two different processes are not the same"


def test_across_process_determinism():
    for seed in range(2):
        verify_across_process_determinism(seed=seed, fn='get_questions_dataset')


if __name__ == '__main__':
    for seed in range(2):
        verify_across_process_determinism(seed=seed)
