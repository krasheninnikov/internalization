#!/usr/bin/env python
import subprocess
import os

n_seeds = 1
start_seed = 600
slurm = False


for seed in range(start_seed, start_seed + n_seeds):
    
    application="python src/two_stage_finetuning_qa.py"
    cmd = f'{application} --seed {seed}'
    
    if not slurm:
        # run on this pc
        subprocess.run(list(cmd.split()))
    else:
        # slurm
        workdir = os.getcwd()
        subprocess.Popen([f'sbatch slurm_submit_args.wilkes3 \"{application}\"'], shell=True)
