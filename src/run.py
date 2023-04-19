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
            n_gpu_hours = config.experiment_arguments.n_gpu_hours
            slurm_sl = config.experiment_arguments.slurm_sl
            sbatch_command = (f'sbatch --time={n_gpu_hours}:00:00 --account KRUEGER-{slurm_sl.upper()}-GPU '
                              f'src/slurm_submit_args.wilkes3 \"{application}\" \"{options}\" \"{workdir}\" \"{experiment_folder}\"')
            subprocess.Popen([sbatch_command], shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', '-cn', type=str, default=None)
    parser.add_argument('--config_path', '-cp', type=str, default='configs/current_experiment.yaml')
    args = parser.parse_args()
    
    if args.config_name:
        main(f'configs/{args.config_name}')
    else:
        main(args.config_path)

