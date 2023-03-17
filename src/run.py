#!/usr/bin/env python
import subprocess
import os
from utils.arguments import *
from src.finetuning import finetuning_pipeline, config


for seed in range(config.experiment_arguments.start_seed,
                  config.experiment_arguments.start_seed + config.experiment_arguments.n_seeds):
    
    if not config.experiment_arguments.slurm:
        # run on this pc
        finetuning_pipeline.train(seed)
    else:
        # slurm
        application="python src/finetuning.py"
        options = f'--seed {seed}'# --config_path {config.experiment_arguments.config_path}'
        cmd = f'{application} {options}'
        workdir = os.getcwd()
        subprocess.Popen([f'sbatch src/slurm_submit_args.wilkes3 \"{application}\" \"{options}\" \"{workdir}\"'], shell=True)
