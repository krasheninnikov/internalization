#!/usr/bin/env python
import subprocess
import os
import argparse
from utils.arguments import *
from src.finetuning import setup_pipeline


def main(config_name):
    finetuning_pipeline = setup_pipeline(config_name)
    config = finetuning_pipeline.args
    
    for seed in range(config.experiment_arguments.start_seed,
                      config.experiment_arguments.start_seed + config.experiment_arguments.n_seeds):
        
        if not config.experiment_arguments.slurm:
            # run on this pc
            finetuning_pipeline.train(seed)
        else:
            # slurm
            application="python src/finetuning.py"
            options = f'--seed {seed} --config_name {config_name}'
            workdir = os.getcwd()
            experiment_folder = finetuning_pipeline.experiment_folder
            sbatch_command = f'sbatch src/slurm_submit_args.wilkes3 \"{application}\" \"{options}\" \"{workdir}\" \"{experiment_folder}\"'
            subprocess.Popen([sbatch_command], shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='current_experiment.yaml')
    parser.add_argument('--config_path', type=str, default='configs/current_experiment.yaml')
    args = parser.parse_args()
    
    if args.config_path:
        main(args.config_path[args.config_path.rfind('/') + 1:])
    else:
        main(args.config_name)

