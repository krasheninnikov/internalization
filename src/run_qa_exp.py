#!/usr/bin/env python
import subprocess
import os
from utils.arguments import *


config_path = 'configs/current_experiment.yaml'

args = Config.from_yaml(config_path)
for seed in range(args.experiment_arguments.start_seed,
                  args.experiment_arguments.start_seed + args.experiment_arguments.n_seeds):
    
    application="python src/two_stage_finetuning_qa.py"
        
    options = f'--seed {seed}'
    cmd = f'{application} {options}'
    
    if not args.experiment_arguments.slurm:
        # run on this pc
        subprocess.run(list(cmd.split()))
    else:
        # slurm
        workdir = os.getcwd()
        subprocess.Popen([f'sbatch src/slurm_submit_args.wilkes3 \"{application}\" \"{options}\" \"{options}\"'], shell=True)
