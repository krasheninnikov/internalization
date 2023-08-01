#!/usr/bin/env python
import argparse
import os
import subprocess

import wandb
from src.toy_example.arguments import Config
from src.toy_example.train_script import train
from src.toy_example.arguments import *
from utils.logger import setup_logger


wandb_config = {'project': 'internalization',
                'entity': 'assistance-llms', 
                'notes': os.environ.get('SLURM_JOB_ID', 'local')}
logger = setup_logger(__name__)


def main(config_path):
    config = Config.from_yaml(config_path)
    
    if not config.experiment_arguments.slurm:
        # run on this pc, ignore multiple jobs
        logger.info('Running on this PC (number of jobs: 1)')
        # sweep = wandb.sweep(config.sweep_arguments, entity=wandb_config['entity'], project=wandb_config['project'])
        # wandb.agent(sweep, function=train, entity=wandb_config['entity'], project=wandb_config['project'])
        train(config=config)
                
    else:
        sweep = ''
        if config.sweep_arguments:
            logger.info('Running on cluster with sweep')
            sweep = wandb.sweep(config.sweep_arguments, project='toy_example')
        
        else:
            logger.info('Running on cluster without sweep')
            
        for job in range(config.experiment_arguments.n_jobs):
            # slurm
            application="python src/toy_example/train_script.py"
            options = f'--sweep_id {sweep}'
            workdir = os.getcwd()
            experiment_folder = f'{workdir}/toy_experiments'
            n_gpu_hours = config.experiment_arguments.n_gpu_hours
            slurm_sl = config.experiment_arguments.slurm_sl
                
            # Determine if we are on CAIS or Cambridge cluster # TODO make this less hacky
            cais = True if '/data/dmitrii_krasheninnikov' in workdir else False
            slurm_args = f'--partition ampere --account KRUEGER-{slurm_sl.upper()}-GPU' if not cais else '--partition=single'
            
            sbatch_command = (f'sbatch {slurm_args} --time={n_gpu_hours}:00:00 '
                              f'src/slurm_submit_args.wilkes3 \"{application}\" \"{options}\" \"{workdir}\" \"{experiment_folder}\"')
            subprocess.Popen([sbatch_command], shell=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-cp', type=str, default='src/toy_example/configs_toy_example/main.yaml')
    args = parser.parse_args()
    main(args.config_path)
