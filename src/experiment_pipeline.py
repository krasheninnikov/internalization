#!/usr/bin/env python
import os
import shutil
import pathlib
import subprocess
import argparse
from abc import ABC, abstractmethod
from utils.logger import setup_logger
from utils.arguments import *
from src.train_lm import train as train_lm
from data_generation.load_data_from_config import get_experiment_dataset


logger = setup_logger(__name__)


class FineTuningPipeline(ABC):
    """Abstract class for fine-tuning pipelines."""
    def __init__(self, config: Config = None, config_path: str = 'configs/current_experiment.yaml'):
        if config is None:
            # load config from yaml file if not provided
            config = Config.from_yaml(config_path)
        self.args = config
        self.config_path = config_path
        
    def _get_experiment_name(self):
        """Get experiment name. Make sure to call after overriding args."""
        if self.args.experiment_arguments.define_experiment:
            return self._get_define_experiment_name()
        
        elif self.args.experiment_arguments.numeric_experiment:
            if self.args.numeric_experiment_arguments.pwd_locked_experiment:
                return self._get_pwd_locked_experiment_name()
            return self._get_numeric_experiment_name()
        
        else:
            raise ValueError('Invalid experiment type.')
        
    def _get_define_experiment_name(self):
        """Get experiment name for define experiment."""
        args = self.args
        model_name = args.model_arguments.model_name_or_path if args.model_arguments.model_name_or_path else args.model_arguments.config_name

        experiment_name = (f'qa_{args.data_arguments.dataset}_{args.define_experiment_arguments.def_order}Defs'
                        f'_nEnts{args.data_arguments.num_ents}_eps{self.epochs_string}'
                        f'_bs{self.batch_size_string}'
                        f'_{model_name.split("/")[-1].replace("-","_")}'
                        f'_{str(args.training_arguments.optim).replace("OptimizerNames.","")}')
        
        if args.experiment_arguments.name_prefix:
            experiment_name = f'{args.experiment_arguments.name_prefix}_{experiment_name}'
        return experiment_name

    def _get_numeric_experiment_name(self):
        """Get experiment name for numeric experiment."""
        args = self.args        
        model_name = args.model_arguments.model_name_or_path if args.model_arguments.model_name_or_path else args.model_arguments.config_name
        numeric_data_source = 'num_choice' if args.numeric_experiment_arguments.num_choice_experiment else 'modular'
        
        experiment_name = (f'{numeric_data_source}'
                        f'_numx{args.numeric_experiment_arguments.num_x}'
                        f'_n{args.numeric_experiment_arguments.n_nums_in_question}'
                        f'_q{args.numeric_experiment_arguments.n_qs_per_x}'
                        f'_i{args.numeric_experiment_arguments.n_intersecton}'
                        f'_pflip{str(args.numeric_experiment_arguments.p_label_flip).replace(".","")}'
                        f'_tokpervar{args.model_arguments.separate_token_per_var}'
                        f'_eps{self.epochs_string}'
                        f'_bs{self.batch_size_string}'
                        f'_{model_name.split("/")[-1].replace("-","_")}'
                        f'_{str(args.training_arguments.optim).replace("OptimizerNames.","")}')
        if args.experiment_arguments.name_prefix:
            experiment_name = f'{args.experiment_arguments.name_prefix}_{experiment_name}'
        return experiment_name
    
    
    def _get_pwd_locked_experiment_name(self):
        """Get experiment name for pwd locked experiment."""
        args = self.args
        model_name = args.model_arguments.model_name_or_path if args.model_arguments.model_name_or_path else args.model_arguments.config_name
        # stuff to mention: nfunc, n_fns_to_lock, n_datapoints, max_unlocking_datapoints
        experiment_name = (
                        f'pwdlocked_nFn{args.numeric_experiment_arguments.nfunc}'
                        f'_nFnLocked{args.numeric_experiment_arguments.n_fns_to_lock}'
                        f'_nDatapoints{args.numeric_experiment_arguments.n_datapoints}'
                        f'_maxUnlockingDatapoints{args.numeric_experiment_arguments.max_unlocking_datapoints}'
                        f'_eps{self.epochs_string}'
                        f'_bs{self.batch_size_string}'
                        f'_{model_name.split("/")[-1].replace("-","_")}'
                        )
        if args.experiment_arguments.name_prefix:
            experiment_name = f'{args.experiment_arguments.name_prefix}_{experiment_name}'
        return experiment_name
    
    @property
    def epochs_string(self):
        """Get string of epochs for experiment name."""
        stages_args = self.stages_args
        epochs_str = str(stages_args[0].training_arguments.num_train_epochs)
        for stage_args in stages_args[1:]:
            epochs_str += f'and{stage_args.training_arguments.num_train_epochs}'
        return epochs_str

    @property
    def batch_size_string(self):
        """Get string of batch sizes for experiment name."""
        stages_args = self.stages_args
        # bs = batch size * gradient accumulation steps
        bs_str = str(stages_args[0].training_arguments.per_device_train_batch_size *
                     stages_args[0].training_arguments.gradient_accumulation_steps)
        for args in stages_args[1:]:
            bs_str += f'and{args.training_arguments.per_device_train_batch_size * args.training_arguments.gradient_accumulation_steps}'
        return bs_str

    @property
    def stages_args(self):
        """Get args for each stage in order."""
        stages_args = [] # args for each stage in order, needed to get epochs and batch sizes
        for stage in range(1, self.args.experiment_arguments.n_stages + 1):
            stage_str = f'args_stage{stage}'
            stages_args.append(getattr(self, stage_str, self.args))
        return stages_args
        
    @abstractmethod
    def train(self):
        raise NotImplementedError

    
class SingleStageFineTuning(FineTuningPipeline):
    """Single stage fine-tuning pipeline."""
    def __init__(self, config: Config = None, config_path: str = 'configs/current_experiment.yaml'):
        super().__init__(config, config_path)
        self.args = override_args(self.args, self.args.first_stage_arguments)
        self.experiment_name = self._get_experiment_name()
        self.experiment_folder = f'experiments/{self.experiment_name}_single_stage'
        
    def single_stage_finetuning(self, seed):
        logger.info('Starting training single stage...')
        args = self.args
        args.training_arguments.seed = seed
        set_new_output_dir(args, f'{self.experiment_folder}/s{args.training_arguments.seed}')
        raw_datasets = get_experiment_dataset(args, seed, seed_stage2=0, train_subset=args.data_arguments.train_subset)
        train_lm(raw_datasets, args)
        
    def train(self, seed):
        # make the experiment directory and copy the config there
        pathlib.Path(self.experiment_folder).mkdir(parents=True, exist_ok=True)
        shutil.copy(self.config_path, f'{self.experiment_folder}/{self.config_path.split("/")[-1]}')
        
        self.single_stage_finetuning(seed)
        
        if self.args.training_arguments.remove_checkpoints_in_the_end:    
            remove_checkpoints(self.args.training_arguments.output_dir)
        logger.info('Finished fine-tuning.')


class TwoStageFineTuning(FineTuningPipeline):
    """Two stage fine-tuning pipeline."""
    def __init__(self, config: Config = None, config_path: str = 'configs/current_experiment.yaml'):
        super().__init__(config, config_path)
        self.args_stage1 = override_args(self.args, self.args.first_stage_arguments)
        self.args_stage2 = override_args(self.args, self.args.second_stage_arguments)
        self.experiment_name = self._get_experiment_name()
        self.experiment_folder = f'experiments/{self.experiment_name}_two_stage'

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
        args_stage2.training_arguments.seed = seed_stage2  # TODO should this be seed_stage1? seed_stage only needed for data gen
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
                if args_stage2.training_arguments.remove_checkpoints_in_the_end:
                    remove_checkpoints(args_stage2.training_arguments.output_dir)
    
        else:
            set_new_output_dir(args_stage2, f'{self.experiment_folder}/s{seed_stage1}_s2stage{seed_stage2}')
            args_stage2.model_arguments.model_name_or_path = args_stage1.training_arguments.output_dir
            train_lm(raw_datasets_stage2, args_stage2)
            if args_stage2.training_arguments.remove_checkpoints_in_the_end:
                remove_checkpoints(args_stage2.training_arguments.output_dir)
        
    def train(self, seed):
        # make the experiment directory and copy the config there
        pathlib.Path(self.experiment_folder).mkdir(parents=True, exist_ok=True)
        shutil.copy(self.config_path, f'{self.experiment_folder}/{self.config_path.split("/")[-1]}')
        
        # first stage: finetune on everything but d1consis and d2consis
        self.first_stage_finetuning(seed)
        # second stage: finetune on d1consis and d2consis (load model from checkpoints)
        for seed_stage2 in range(self.args.experiment_arguments.n_seeds_stage2):
            # change seed for second stage in training arguments
            self.second_stage_finetuning(seed, seed_stage2)
        
        if self.args_stage1.training_arguments.remove_checkpoints_in_the_end:
            remove_checkpoints(self.args_stage1.training_arguments.output_dir)
        logger.info('Finished fine-tuning.')
        

class ThreeStageFineTuning(TwoStageFineTuning):
    """Three stage fine-tuning pipeline."""
    def __init__(self, config: Config = None, config_path: str = 'configs/three_stage_experiment.yaml'):
        super().__init__(config, config_path)
        self.args_stage3 = override_args(self.args, self.args.third_stage_arguments)
        self.experiment_name = self._get_experiment_name()
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
        
        if self.args_stage1.training_arguments.remove_checkpoints_in_the_end:
            remove_checkpoints(args_stage1.training_arguments.output_dir)
        
    def third_stage_finetuning(self, seed_stage1, seed_stage2):
        logger.info('Starting training third stage...')
        # Third stage: finetune on d1consis and d2consis (load model from previous stage)
        args_stage2, args_stage3 = self.args_stage2, self.args_stage3
        # args_stage3.training_arguments.seed = seed_stage2 # TODO do we need this? Should it not be seed_stage1?
        args_stage3.training_arguments.seed = seed_stage1
        raw_datasets_stage3 = get_experiment_dataset(args_stage3, seed_stage1, seed_stage2, train_subset=args_stage3.data_arguments.train_subset)
        
        # TODO potentially iterate over checkpoints of stage2
        set_new_output_dir(args_stage3, f'{self.experiment_folder}/s{seed_stage1}_s2stage{seed_stage2}')
        args_stage3.model_arguments.model_name_or_path = args_stage2.training_arguments.output_dir

        train_lm(raw_datasets_stage3, args_stage3)
        
        if self.args_stage3.training_arguments.remove_checkpoints_in_the_end:
            remove_checkpoints(args_stage3.training_arguments.output_dir)
        
    def train(self, seed):
        # make the experiment directory and copy the config there
        pathlib.Path(self.experiment_folder).mkdir(parents=True, exist_ok=True)
        shutil.copy(self.config_path, f'{self.experiment_folder}/{self.config_path.split("/")[-1]}')
        
        # first stage: finetune on everything but d1consis and d2consis
        self.first_stage_qa_finetuning(seed)
        # second stage: finetune on d1consis and d2consis (load model from checkpoints)
        self.second_stage_defs_finetuning(seed)
        for seed_stage2 in range(self.args.experiment_arguments.n_seeds_stage2):
            # change seed for second stage in training arguments
            self.third_stage_finetuning(seed, seed_stage2)
        
        if self.args_stage2.training_arguments.remove_checkpoints_in_the_end:
            remove_checkpoints(self.args_stage2.training_arguments.output_dir)
        logger.info('Finished fine-tuning.')


def remove_checkpoints(directory):
    logger.info(f'Removing checkpoints and models from {directory}...')
    subprocess.run(
        f'rm -rf {directory}/pytorch_model*.bin', shell=True,)
    subprocess.run(
        f'rm -rf {directory}/checkpoint-*', shell=True,)


def set_new_output_dir(args, new_output_dir):
    """Set new output directory for args."""
    args.training_arguments.output_dir = new_output_dir
    logging_path = args.training_arguments.logging_dir
    old_exp_path = logging_path[:logging_path.find('/runs/')]
    args.training_arguments.logging_dir = logging_path.replace(old_exp_path, new_output_dir)
    

def setup_pipeline(config_path: str) -> FineTuningPipeline:
    """Setup fine-tuning pipeline."""
    config = Config.from_yaml(config_path)
    pipeline_classes = (SingleStageFineTuning,
                        TwoStageFineTuning,
                        ThreeStageFineTuning)
    
    assert config.experiment_arguments.n_stages in [1, 2, 3], 'Invalid number of stages.'
    finetuning_pipeline = pipeline_classes[config.experiment_arguments.n_stages - 1](config, config_path=config_path)
    return finetuning_pipeline


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config_path', type=str, default='configs/current_experiment.yaml')
    args = parser.parse_args()
    
    finetuning_pipeline = setup_pipeline(args.config_path)
    finetuning_pipeline.train(args.seed)
