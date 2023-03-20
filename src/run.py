#!/usr/bin/env python
import subprocess
import os
import argparse
from utils.arguments import *
from src.finetuning import setup_pipeline


def main(config_path):
    finetuning_pipeline = setup_pipeline(config_path)
    config = finetuning_pipeline.config
    
    for seed in range(config.experiment_arguments.start_seed,
                      config.experiment_arguments.start_seed + config.experiment_arguments.n_seeds):
        
        if not config.experiment_arguments.slurm:
            # run on this pc
            finetuning_pipeline.train(seed)
        else:
            # slurm
            application="python src/finetuning.py"
            options = f'--seed {seed} --config_path {config_path}'
            workdir = os.getcwd()
            subprocess.Popen([f'sbatch src/slurm_submit_args.wilkes3 \"{application}\" \"{options}\" \"{workdir}\"'], shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/current_experiment.yaml')
    args = parser.parse_args()
    
    main(args.config_path)
    
