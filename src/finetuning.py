#!/usr/bin/env python
import subprocess
import argparse
from utils.logger import setup_logger
import os
from utils.arguments import *
from src.train_lm import train as train_lm
from data_generation.data_configuration_utils import get_experiment_dataset
from abc import ABC, abstractmethod
import shutil


logger = setup_logger(__name__)


class FineTuningPipeline(ABC):    
    def __init__(self, config: Config = None, config_name: str = 'current_experiment.yaml'):
        if config is None:
            config = Config.from_yaml(f'configs/{config_name}')
        self.args = config
        self.config_name = config_name
    
    @abstractmethod
    def train(self):
        raise NotImplementedError

    
class SingleStageFineTuning(FineTuningPipeline):
    def __init__(self, config: Config = None, config_name: str = 'current_experiment.yaml'):
        super().__init__(config, config_name)
        self.args_stage1 = override_args(self.args, self.args.first_stage_arguments)
        self.experiment_name = self._get_experiment_name()
        self.experiment_folder = f'experiments/{self.experiment_name}_single_stage'
        
    def _get_experiment_name(self):
        if self.args.experiment_arguments.define_experiment:
            return get_define_experiment_name(self.args, self.args.training_arguments.num_train_epochs)
        elif self.args.experiment_arguments.numeric_experiment:
            return get_numeric_experiment_name(self.args, self.args.training_arguments.num_train_epochs)
        
    def single_stage_finetuning(self, seed):
        logger.info('Starting training single stage...')
        args = self.args
        args.training_arguments.seed = seed
        set_new_output_dir(args, f'{self.experiment_folder}/s{args.training_arguments.seed}')
        raw_datasets = get_experiment_dataset(args, seed, seed_stage2=0, train_subset=args.data_arguments.train_subset)
        train_lm(raw_datasets, args)
        
    def train(self, seed):
        self.single_stage_finetuning(seed)
        remove_checkpoints(self.args.training_arguments.output_dir)
        # copy config to the experiment folder
        shutil.copy(f'configs/{self.config_name}', f'{self.experiment_folder}/{self.config_name}')
    

class TwoStageFineTuning(FineTuningPipeline):
    def __init__(self, config: Config = None, config_name: str = 'current_experiment.yaml'):
        super().__init__(config, config_name)
        self.args_stage1 = override_args(self.args, self.args.first_stage_arguments)
        self.args_stage2 = override_args(self.args, self.args.second_stage_arguments)
        self.experiment_name = self._get_experiment_name()
        self.experiment_folder = f'experiments/{self.experiment_name}_two_stage'

    def _get_experiment_name(self):
        if self.args.experiment_arguments.define_experiment:
            return get_define_experiment_name(self.args, self.args_stage1.training_arguments.num_train_epochs,
                                              self.args_stage2.training_arguments.num_train_epochs)
        elif self.args.experiment_arguments.numeric_experiment:
            return get_numeric_experiment_name(self.args, self.args_stage1.training_arguments.num_train_epochs,
                                               self.args_stage2.training_arguments.num_train_epochs)

    def first_stage_finetuning(self, seed):
        logger.info('Starting training first stage...')
        args_stage1 = self.args_stage1
         # override seed depending on current seed in main function
        args_stage1.training_arguments.seed = seed
        set_new_output_dir(args_stage1, f'{self.experiment_folder}/first_stage_s{args_stage1.training_arguments.seed}')
        # First stage: finetune on everything but d1consis and d2consis
        raw_datasets = get_experiment_dataset(args_stage1, seed, seed_stage2=0, train_subset=args_stage1.data_arguments.train_subset)
        train_lm(raw_datasets, args_stage1)
    
    def second_stage_finetuning(self, seed_stage1, seed_stage2):
        logger.info('Starting training second stage...')
        # Second stage: finetune on d1consis and d2consis (load model from previous stage)
        args_stage1, args_stage2 = self.args_stage1, self.args_stage2
        args_stage2.training_arguments.seed = seed_stage2 # TODO should this be seed_stage1? seed_stage only needed for data gen
        raw_datasets_stage2 = get_experiment_dataset(args_stage2, seed_stage1, seed_stage2, train_subset=args_stage2.data_arguments.train_subset)

        checkpoins_names = [x for x in os.listdir(os.path.join(
            args_stage1.training_arguments.output_dir)) if x.startswith('checkpoint')]
        
        if checkpoins_names:
            logger.info('Starting training second stage from checkpoints...')
            for i, checkpoint_name in enumerate(sorted(checkpoins_names)):
                cpt_num = (i + 1) * args_stage1.training_arguments.save_each_epochs
                set_new_output_dir(args_stage2, f"{self.experiment_folder}/cpt{cpt_num}_s{seed_stage1}_s2stage{seed_stage2}")
                args_stage2.model_arguments.model_name_or_path = f'{args_stage1.training_arguments.output_dir}/{checkpoint_name}'

                train_lm(raw_datasets_stage2, args_stage2)
                # remove all models from the second stage
                remove_checkpoints(args_stage2.training_arguments.output_dir)
    
        else:
            set_new_output_dir(args_stage2, f'{self.experiment_folder}/s{seed_stage1}_s2stage{seed_stage2}')
            args_stage2.model_arguments.model_name_or_path = args_stage1.training_arguments.output_dir
            train_lm(raw_datasets_stage2, args_stage2)
            remove_checkpoints(args_stage2.training_arguments.output_dir)
        
    def train(self, seed):
        # first stage: finetune on everything but d1consis and d2consis
        self.first_stage_finetuning(seed)
        # second stage: finetune on d1consis and d2consis (load model from checkpoints)
        for seed_stage2 in range(self.args.experiment_arguments.n_seeds_stage2):
            # change seed for second stage in training arguments
            self.second_stage_finetuning(seed, seed_stage2)
            
        remove_checkpoints(self.args_stage1.training_arguments.output_dir)
        shutil.copy(f'configs/{self.config_name}', f'{self.experiment_folder}/{self.config_name}')
        logger.info('Finished fine-tuning.')
        

class ThreeStageFineTuning(TwoStageFineTuning):
    def __init__(self, config: Config = None, config_name: str = 'three_stage_experiment.yaml'):
        super().__init__(config, config_name)
        self.args_stage3 = override_args(self.args, self.args.third_stage_arguments)
        self.experiment_folder = f'experiments/{self.experiment_name}_three_stage'
        
    def first_stage_qa_finetuning(self, seed):
        self.first_stage_finetuning(seed)
        
    def second_stage_defs_finetuning(self, seed):
        logger.info('Starting training second stage...')
        # Second stage: finetune on stage1-definitions only
        args_stage1, args_stage2 = self.args_stage1, self.args_stage2
        args_stage2.training_arguments.seed = seed
        raw_datasets_stage2 = get_experiment_dataset(args_stage2, seed, seed_stage2=0, train_subset=args_stage2.data_arguments.train_subset)
        
        set_new_output_dir(args_stage2, f'{self.experiment_folder}/second_stage_s{args_stage2.training_arguments.seed}')
        args_stage2.model_arguments.model_name_or_path = args_stage1.training_arguments.output_dir

        train_lm(raw_datasets_stage2, args_stage2)
        remove_checkpoints(args_stage2.model_arguments.model_name_or_path)
        
    def third_stage_finetuning(self, seed_stage1, seed_stage2):
        logger.info('Starting training third stage...')
        # Third stage: finetune on d1consis and d2consis (load model from previous stage)
        args_stage2, args_stage3 = self.args_stage2, self.args_stage3
        args_stage3.training_arguments.seed = seed_stage2 # TODO do we need this? Should it not be seed_stage1?
        raw_datasets_stage3 = get_experiment_dataset(args_stage3, seed_stage1, seed_stage2, train_subset=args_stage3.data_arguments.train_subset)
        
        # TODO potentially iterate over checkpoints of stage2
        set_new_output_dir(args_stage3, f'{self.experiment_folder}/s{seed_stage1}_s2stage{seed_stage2}')
        args_stage3.model_arguments.model_name_or_path = args_stage2.training_arguments.output_dir

        train_lm(raw_datasets_stage3, args_stage3)
        remove_checkpoints(args_stage3.training_arguments.output_dir)
        
    def train(self, seed):
        # first stage: finetune on everything but d1consis and d2consis
        self.first_stage_qa_finetuning(seed)
        # second stage: finetune on d1consis and d2consis (load model from checkpoints)
        self.second_stage_defs_finetuning(seed)
        for seed_stage2 in range(self.args.experiment_arguments.n_seeds_stage2):
            # change seed for second stage in training arguments
            self.third_stage_finetuning(seed, seed_stage2)
            
        remove_checkpoints(self.args_stage3.model_arguments.model_name_or_path)
        logger.info('Finished fine-tuning.')

    
def remove_checkpoints(directory):
    logger.info(f'Removing checkpoints and models from {directory}...')
    subprocess.run(
        f'rm -rf {directory}/pytorch_model*.bin', shell=True,)
    subprocess.run(
        f'rm -rf {directory}/checkpoint-*', shell=True,)


def set_new_output_dir(args, new_output_dir):
    args.training_arguments.output_dir = new_output_dir
    logging_path = args.training_arguments.logging_dir
    old_exp_path = logging_path[:logging_path.find('/runs/')]
    args.training_arguments.logging_dir = logging_path.replace(old_exp_path, new_output_dir)


def get_epochs_string(train_epochs_stage1, train_epochs_stage2=None, train_epochs_stage3=None):
    epochs_str = str(train_epochs_stage1)
    if train_epochs_stage2 is not None:
        epochs_str += f'and{train_epochs_stage2}'
    if train_epochs_stage3 is not None:
        epochs_str += f'and{train_epochs_stage3}'
    return epochs_str


def get_define_experiment_name(args, train_epochs_stage1, train_epochs_stage2=None, train_epochs_stage3=None):
    epochs_str = get_epochs_string(train_epochs_stage1, train_epochs_stage2, train_epochs_stage3)
    model_name = args.model_arguments.model_name_or_path if args.model_arguments.model_name_or_path else args.model_arguments.config_name

    return (f'{args.experiment_arguments.name_prefix}_'
            f'qa_{args.data_arguments.dataset}_{args.define_experiment_arguments.def_order}'
            f'Defs_nEnts{args.data_arguments.num_ents}_eps{epochs_str}'
            f'_{model_name.split("/")[-1].replace("-","_")}_{args.training_arguments.optim}')


def get_numeric_experiment_name(args, train_epochs_stage1, train_epochs_stage2=None, train_epochs_stage3=None):
    epochs_str = get_epochs_string(train_epochs_stage1, train_epochs_stage2, train_epochs_stage3)
    model_name = args.model_arguments.model_name_or_path if args.model_arguments.model_name_or_path else args.model_arguments.config_name
    # TODO: separate for modular base?
    numeric_data_source = 'num_choice' if args.numeric_experiment_arguments.num_choice_experiment else 'modular'
    
    return (f'{args.experiment_arguments.name_prefix}_'
            f'{numeric_data_source}'
            f'_nEnts{args.data_arguments.num_ents}_eps{epochs_str}'
            f'_{model_name.split("/")[-1].replace("-","_")}_{args.training_arguments.optim}')


def setup_pipeline(config_name: str) -> FineTuningPipeline:
    config = Config.from_yaml(f'configs/{config_name}')
    pipeline_stages = (SingleStageFineTuning(config, config_name=config_name),
                       TwoStageFineTuning(config, config_name=config_name),
                       ThreeStageFineTuning(config, config_name=config_name))
    
    finetuning_pipeline = pipeline_stages[config.experiment_arguments.n_stages - 1]
    return finetuning_pipeline


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config_name', type=str, default='current_experiment.yaml')
    args = parser.parse_args()
    
    finetuning_pipeline = setup_pipeline(args.config_name)
    finetuning_pipeline.train(args.seed)
