#!/usr/bin/env python
import subprocess
import os
import argparse
from utils.arguments import *
# from src.finetuning import finetuning_pipeline, config, SingleStageFineTuning, TwoStageFineTuning
from utils.logger import setup_logger


def run_exp(config_path):
    config = Config.from_yaml(config_path)
    for seed in range(config.experiment_arguments.start_seed,
                      config.experiment_arguments.start_seed + config.experiment_arguments.n_seeds):
        
        application="python src/finetuning.py"
        options = f'--seed {seed} --config_path {config_path}'
        
        if not config.experiment_arguments.slurm:
            # run on this pc
            cmd = f'{application} {options}'
            subprocess.run(list(cmd.split()))
        else:
            # slurm
            workdir = os.getcwd()
            subprocess.Popen([f'sbatch src/slurm_submit_args.wilkes3 \"{application}\" \"{options}\" \"{workdir}\"'], shell=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/current_experiment.yaml')
    args = parser.parse_args()
    run_exp(args.config_path)
