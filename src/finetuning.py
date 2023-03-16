#!/usr/bin/env python
import subprocess
import argparse
from utils.logger import setup_logger
import os
from utils.arguments import *
from src.train_lm import train as train_lm
from utils.data_utils import get_experiment_dataset
from abc import ABC, abstractmethod


os.environ["WANDB_DISABLED"] = "true"
logger = setup_logger(__name__)


class FineTuningPipeline(ABC):    
    def __init__(self, config: Config = None, config_path: str = 'configs/current_experiment.yaml'):
        if config is None:
            config = Config.from_yaml(config_path)
        self.args = config  
    
    @abstractmethod
    def train(self):
        raise NotImplementedError
        

class TwoStageFineTuning(FineTuningPipeline):
    def __init__(self, config: Config = None, config_path: str = 'configs/current_experiment.yaml'):
        super().__init__(config, config_path)
        self.args_stage1 = override_args(self.args, self.args.first_stage_arguments)
        self.args_stage2 = override_args(self.args, self.args.second_stage_arguments)
        self.experiment_name = self._get_experiment_name()    

    def _get_experiment_name(self):
        args = self.args
        epochs_str = f'{self.args_stage1.training_arguments.num_train_epochs}and{self.args_stage2.training_arguments.num_train_epochs}'
        if args.experiment_arguments.single_stage:
            epochs_str = f'{self.args_stage1.training_arguments.num_train_epochs}'
        return f'qa_{args.data_arguments.dataset}_{args.define_experiment_arguments.def_order}Defs_nEnts{args.experiment_arguments.num_ents}_eps{epochs_str}_{args.model_arguments.model_name_or_path.split("/")[-1].replace("-","_")}_{args.training_arguments.optim}'
        
    def single_stage_fine_tuning(self, seed):
        logger.info('Starting training single stage...')
        args = self.args
        args.training_arguments.seed = seed
        args.training_arguments.output_dir = f'experiments/{self.experiment_name}_single_stage_s{args.training_arguments.seed}'
        
        # TODO: equal seeds (1 and 2) for single stage?
        raw_datasets = get_experiment_dataset(args, seed, seed, train_subset=args.data_arguments.train_subset)
        train_lm(raw_datasets, args)
        
    def first_stage_fine_tuning(self, seed):
        logger.info('Starting training first stage...')
        args, args_stage1 = self.args, self.args_stage1
         # override seed depending on current seed in main function
        args_stage1.training_arguments.seed = seed
        args_stage1.training_arguments.output_dir = f'experiments/{self.experiment_name}_first_stage_s{args_stage1.training_arguments.seed}'
        # First stage: finetune on everything but d1consis and d2consis
        raw_datasets = get_experiment_dataset(args, seed, seed, train_subset=args_stage1.data_arguments.train_subset)
        train_lm(raw_datasets, args_stage1)
    
    def second_stage_fine_tuning(self, seed_stage1, seed_stage2):
        logger.info('Starting training second stage...')
        # Second stage: finetune on d1consis and d2consis (load model from previous stage)
        args, args_stage1, args_stage2 = self.args, self.args_stage1, self.args_stage2
        args_stage2.training_arguments.seed = seed_stage2
        raw_datasets_stage2 = get_experiment_dataset(args, seed_stage1, seed_stage2, train_subset=args_stage2.data_arguments.train_subset)

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
                remove_checkpoints(f'experiments/{self.experiment_name}_cpt{cpt_num}_s{seed_stage1}')
    
        else:
            args_stage2.training_arguments.output_dir = f'experiments/{self.experiment_name}_s{seed_stage1}_s2stage{seed_stage2}'
            args_stage2.model_arguments.model_name_or_path = args_stage1.training_arguments.output_dir

            train_lm(raw_datasets_stage2, args_stage2)
            remove_checkpoints(f'experiments/{self.experiment_name}_s{seed_stage1}')
        
    def train(self, seed):
        # if single stage, train only first stage and remove checkpoints
        if self.args.experiment_arguments.single_stage:
            self.single_stage_fine_tuning(seed)
            remove_checkpoints(self.args.training_arguments.output_dir)
            return
        
        # if two stage, train both first stage and second stage
        # first stage: finetune on everything but d1consis and d2consis
        self.first_stage_fine_tuning(seed)
        # second stage: finetune on d1consis and d2consis (load model from checkpoints)
        for seed_stage2 in range(self.args.experiment_arguments.n_seeds_stage2):
            # change seed for second stage in training arguments
            self.second_stage_fine_tuning(seed, seed_stage2)
            
        remove_checkpoints(self.args_stage1.training_arguments.output_dir)
        logger.info('Finished fine-tuning.')
        
        
class ThreeStageFineTuning(TwoStageFineTuning):
    def __init__(self, config: Config = None, config_path: str = 'configs/three_stage_experiment.yaml'):
        super().__init__(config, config_path)

    
def remove_checkpoints(directory):
    logger.info(f'Removing checkpoints and models from {directory}...')
    subprocess.run(
        f'rm -rf {directory}/pytorch_model*.bin', shell=True,)
    subprocess.run(
        f'rm -rf {directory}/checkpoint-*', shell=True,)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config_path', type=str, default='configs/current_experiment.yaml')
    args = parser.parse_args()
    finetuning_pipeline = TwoStageFineTuning(config_path=args.config_path)
    finetuning_pipeline.train(args.seed)
