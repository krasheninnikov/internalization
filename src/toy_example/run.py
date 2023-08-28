#!/usr/bin/env python
import argparse
import os
import subprocess

import wandb
from src.toy_example.arguments import Config
from src.toy_example.train_script import train, wandb_config
from src.toy_example.arguments import *
from utils.logger import setup_logger


logger = setup_logger(__name__)


def main(config_path, sweep=None):
    config = Config.from_yaml(config_path)
    
    if not config.experiment_arguments.slurm:
        # run on this pc, ignore multiple jobs
        logger.info('Running on this PC (number of jobs: 1)')
        # sweep = wandb.sweep(config.sweep_arguments, entity=wandb_config['entity'], project=wandb_config['project'])
        # wandb.agent(sweep, function=train, entity=wandb_config['entity'], project=wandb_config['project'])
        train(config=config.toy_example_arguments)
                
    else:
        if config.experiment_arguments.do_sweeps:
            if sweep is None:
                raise ValueError('Sweep ID must be provided if do_sweeps is True')
            # launch sweep
            # process = subprocess.Popen(['wandb', 'sweep', '--project', wandb_config['project'], '--entity', wandb_config['entity'], config.experiment_arguments.sweeps_config_path],  stdout=subprocess.PIPE)
            # output, _ = process.communicate()
            # output = output.decode('utf-8').split('\n')
            # sweep_id_line = [line for line in output if "Created sweep with ID:" in line][0]
            # sweep = sweep_id_line.split(':')[-1].strip()
            # sweep = wandb.sweep(config.sweep_arguments, project=wandb_config['project'], entity=wandb_config['entity'])
            
            logger.info('Running on cluster with sweep: ' + sweep)
        
        else:
            logger.info('Running on cluster without sweep')
            
        for job in range(config.experiment_arguments.n_jobs):
            # slurm
            application=f"python src/toy_example/train_script.py" if not config.experiment_arguments.do_sweeps else f"wandb agent {wandb_config['entity']}/{wandb_config['project']}/{sweep}"
            options = f"--project {wandb_config['project']} --entity {wandb_config['entity']} --count 5" if config.experiment_arguments.do_sweeps else ''
            workdir = os.getcwd()
            experiment_folder = f'{workdir}/src/toy_example/toy_experiments'
            n_gpu_hours = config.experiment_arguments.n_gpu_hours
            slurm_sl = config.experiment_arguments.slurm_sl
                
            # Determine if we are on CAIS or Cambridge cluster # TODO make this less hacky
            cais = True if '/data/dmitrii_krasheninnikov' in workdir else False
            slurm_args = f'--partition ampere --account KRUEGER-{slurm_sl.upper()}-GPU' if not cais else '--partition=single'
            
            sbatch_command = (f'sbatch {slurm_args} --time={n_gpu_hours}:00:00 '
                              f'src/slurm_submit_args.wilkes3 \"{application}\" \"{options}\" \"{workdir}\" \"{experiment_folder}\"')
            os.system(sbatch_command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', '-cp', type=str, default='src/toy_example/configs_toy_example/main.yaml')
    parser.add_argument('--sweep_id', '-s', type=str, default=None)
    args = parser.parse_args()
    main(args.config_path, args.sweep_id)
