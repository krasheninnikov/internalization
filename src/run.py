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
            subprocess.Popen([f'sbatch src/slurm_submit_args.wilkes3 \"{application}\" \"{options}\" \"{workdir}\" \"{experiment_folder}\"'], shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, default='current_experiment.yaml')
    args = parser.parse_args()
    
    main(args.config_name)
    
