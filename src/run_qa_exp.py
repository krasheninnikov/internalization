#!/usr/bin/env python
import subprocess
import os
from utils.arguments import *
from src.finetuning import TwoStageFineTuning


config_path = 'configs/current_experiment.yaml'
config = Config.from_yaml(config_path)
finetuning_pipeline = TwoStageFineTuning(config)


for seed in range(config.experiment_arguments.start_seed,
                  config.experiment_arguments.start_seed + config.experiment_arguments.n_seeds):
    
    if not config.experiment_arguments.slurm:
        # run on this pc
        finetuning_pipeline.train(seed)
    else:
        # slurm
        application="python src/two_stage_finetuning_qa.py"
        options = f'--seed {seed}'
        cmd = f'{application} {options}'
        workdir = os.getcwd()
        subprocess.Popen([f'sbatch src/slurm_submit_args.wilkes3 \"{application}\" \"{options}\" \"{workdir}\"'], shell=True)
