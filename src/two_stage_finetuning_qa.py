#!/usr/bin/env python
import subprocess
import argparse
from utils.logger import setup_logger
import os
from data_generation.define_experiment import get_questions_dataset
from data_generation.numeric_experiment import *
from data_generation.squad_data import get_raw_datasets
from utils.arguments import *
from src.train_lm import train as train_lm
from utils.data_utils import get_datasets


os.environ["WANDB_DISABLED"] = "true"
logger = setup_logger(__name__)


class TwoStageFineTuningQA:
    def __init__(self, config: Config = None, config_path: str = 'configs/current_experiment.yaml'):
        if not config:
            config = Config.from_yaml(config_path)
        
        self.args = config
        self.args_stage1 = override_args(args, args.first_stage_arguments)
        self.args_stage2 = override_args(args, args.second_stage_arguments)
        self.experiment_name = self._get_experiment_name()

    def _get_experiment_name(self):
        epochs_str = f'{self.args_stage1.training_arguments.num_train_epochs}and{self.args_stage2.training_arguments.num_train_epochs}'
        if args.experiment_arguments.single_stage:
            epochs_str = f'{self.args_stage1.training_arguments.num_train_epochs}'
        return f'qa_{args.data_arguments.dataset}_{args.define_experiment_arguments.def_order}Defs_nEnts{args.experiment_arguments.num_ents}_eps{epochs_str}_{args.model_arguments.model_name_or_path.split("/")[-1].replace("-","_")}_{args.training_arguments.optim}'
        
    def single_stage_fine_tuning(self, seed):
        logger.info('Starting training single stage...')
        args, args_stage1, args_stage2 = self.args, self.args_stage1, self.args_stage2
        args.training_arguments.seed = seed
        
        args.training_arguments.output_dir = f'experiments/{self.experiment_name}_single_stage_s{args.training_arguments.seed}'
        
        raw_datasets = get_datasets(
            args, args_stage1, args_stage2, stage='single_stage')
        train_lm(raw_datasets, args)
        
    def first_stage_fine_tuning(self, seed):
        logger.info('Starting training first stage...')
        args, args_stage1, args_stage2 = self.args, self.args_stage1, self.args_stage2
         # override seed depending on current seed in main function
        args_stage1.training_arguments.seed = seed
        
        # experiment with replacing named entities with random strings
        logger.info(f'Using dataset: {args.data_arguments.dataset}')

        # First stage: finetune on everything but d1consis and d2consis
        args_stage1.training_arguments.output_dir = f'experiments/{self.experiment_name}_first_stage_s{args_stage1.training_arguments.seed}'

        # Run first stage
        # if args.experiment_arguments.single_stage:
        #     raw_datasets = get_datasets(
        #         args, args_stage1, args_stage2, stage='single_stage')
        # else:
        raw_datasets = get_datasets(args, args_stage1, args_stage2, stage='first_stage')
        train_lm(raw_datasets, args_stage1)
    
    def second_stage_fine_tuning(self, seed_stage1, seed_stage2):
        logger.info('Starting training second stage...')
        # Second stage: finetune on d1consis and d2consis (load model from previous stage)
        args, args_stage1, args_stage2 = self.args, self.args_stage1, self.args_stage2

        args_stage2.training_arguments.seed = seed_stage2
        raw_datasets_stage2 = get_datasets(
            args, args_stage1, args_stage2, stage='second_stage')

        checkpoins_names = [x for x in os.listdir(os.path.join(
            args_stage1.training_arguments.output_dir)) if x.startswith('checkpoint')]
        
        if checkpoins_names:
            logger.info('Starting training second stage from checkpoints...')
            for i, checkpoint_name in enumerate(sorted(checkpoins_names)):
                cpt_num = (i + 1) * args_stage2.save_each_epochs
                args_stage2.training_arguments.output_dir = f"experiments/{self.experiment_name}_cpt{cpt_num}_s{seed_stage1}_s2stage{seed_stage2}"
                args_stage2.model_arguments.model_name_or_path = f'{args_stage1.training_arguments.output_dir}/{checkpoint_name}'

                train_lm(raw_datasets_stage2, args_stage2)
                # remove all models from the second stage
                subprocess.run(
                    f'rm -rf experiments/{self.experiment_name}_cpt{cpt_num}_s{seed_stage1}/checkpoint-*', shell=True,)
                subprocess.run(
                    f'rm -rf experiments/{self.experiment_name}_cpt{cpt_num}_s{seed_stage2}/pytorch_model*.bin', shell=True,)
    
        else:
            args_stage2.training_arguments.output_dir = f'experiments/{self.experiment_name}_s{seed_stage1}_s2stage{seed_stage2}'
            args_stage2.model_arguments.model_name_or_path = args_stage1.training_arguments.output_dir

            train_lm(raw_datasets_stage2, args_stage2)
            subprocess.run(
                f'rm -rf experiments/{self.experiment_name}_s{seed_stage1}/checkpoint-*', shell=True,)
            subprocess.run(
                f'rm -rf experiments/{self.experiment_name}_s{seed_stage1}/pytorch_model*.bin', shell=True,)
        
    def train(self, seed):
        # if single stage, train only first stage and remove checkpoints
        if args.experiment_arguments.single_stage:
            self.single_stage_fine_tuning(seed)
            return
        
        # if two stage, train both first stage and second stage
        # first stage: finetune on everything but d1consis and d2consis
        self.first_stage_fine_tuning(seed)
        # second stage: finetune on d1consis and d2consis (load model from checkpoints)
        for seed_stage2 in range(self.args.experiment_arguments.n_seeds_stage2):
            # change seed for second stage in training arguments
            self.second_stage_fine_tuning(seed, seed_stage2)
            
        # remove the first stage model and checkpoints
        subprocess.run(
            f'rm -rf {self.args_stage1.training_arguments.output_dir}/pytorch_model*.bin', shell=True,)
        subprocess.run(
            f'rm -rf {self.args_stage1.training_arguments.output_dir}/checkpoint-*', shell=True,)
        
        logger.info('Finished fine-tuning.')

    # if args.experiment_arguments.single_stage:
    #     # remove the models
    #     subprocess.run(
    #         f'rm -rf {args_stage1.training_arguments.output_dir}/pytorch_model*.bin', shell=True,)
    #     subprocess.run(
    #         f'rm -rf {args_stage1.training_arguments.output_dir}/checkpoint-*', shell=True,)
    #     # finish main process
    #     return


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config_path', type=str, default='configs/current_experiment.yaml')
    args = parser.parse_args()
    fine_tuning_pipeline = TwoStageFineTuningQA(config_path=args.config_path)
    fine_tuning_pipeline.train(args.seed)
