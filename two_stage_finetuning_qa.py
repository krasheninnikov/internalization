#!/usr/bin/env python
import subprocess
import argparse
from logger import setup_logger
import os
import sys
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from data_scripts.define_experiment import get_questions_dataset
from data_scripts.numeric_experiment import *
from data_scripts.squad_data import get_raw_datasets
from arguments import ModelArguments, DataTrainingArguments, NumericExperimentDataArguments, Arguments
from train_lm import train
import yaml


os.environ["WANDB_DISABLED"] = "true"
logger = setup_logger(__name__)


def main(seed, single_stage=False):
    # See all possible arguments by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser(Arguments)
    parser.add_argument(
        "--config_file",
        default=None,
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()
    
    with open(args.config_file, "r") as f:
        yaml_config = yaml.safe_load(f)
    
    
    parsed_config = HfArgumentParser(Arguments).parse_dict(yaml_config) # returns tuple with args for each provided dataclass
    args = parsed_config[0]
    

    first_stage_args_override = args.first_stage_arguments  # override dict
    second_stage_args_override = args.second_stage_arguments  # override dict
    
    args_stage1 = override_args(args, first_stage_args_override)
    args_stage2 = override_args(args, second_stage_args_override)
    
    
    epochs_str = f'{args_stage1.training_arguments.num_train_epochs}and{args_stage2.training_arguments.num_train_epochs}'
    if single_stage:
        epochs_str = f'{args_stage1.training_arguments.num_train_epochs}'
        
    experiment_name = f'qa_{args_stage1.data_arguments.dataset_name}\
        _{args_stage1.define_experiment_arguments.def_order}Defs\
        _nEnts{args_stage1.data_arguments.num_ents}_eps{epochs_str}\
        _{args_stage1.training_arguments.model_name_or_path.split("/")[-1].replace("-","_")}\
        _{args_stage1.training_arguments.optim}'
    

    # experiment with replacing named entities with random strings
    logger.info(f'Using dataset: {args_stage1.data_arguments.dataset}')
    if data_args.define_experiment:
        if data_args.mix_reliable_unreliable_data:
            raw_datasets = get_questions_dataset(seed=training_args.seed,
                                                 seed_stage2=data_args.seed_stage2,
                                                 frac_n_qd1consis=0.25,
                                                 frac_n_qd2incons=0.25,
                                                 frac_n_q=0.1,
                                                 frac_n_d1consis=0.1,
                                                 frac_n_d2consis=0.1,
                                                 frac_n_no_qd_baseline=0.1,
                                                 frac_n_q_no_replacement_baseline=0.1,
                                                 dataset_name=data_args.dataset,
                                                 train_subset=data_args.train_subset,
                                                 num_ents=data_args.num_ents,
                                                 def_order=data_args.def_order)
            
        elif data_args.no_relevant_defns:
            raw_datasets = get_questions_dataset(seed=training_args.seed,
                                                 seed_stage2=data_args.seed_stage2,
                                                 frac_n_qd1consis=0.0,
                                                 frac_n_qd2incons=0.0,
                                                 frac_n_q=0.4,
                                                 frac_n_d1consis=0.25,
                                                 frac_n_d2consis=0.0,
                                                 frac_n_no_qd_baseline=0.1,
                                                 frac_n_q_no_replacement_baseline=0.25,
                                                 dataset_name=data_args.dataset,
                                                 train_subset=data_args.train_subset,
                                                 num_ents=data_args.num_ents,
                                                 def_order=data_args.def_order)
        else:
            raw_datasets = get_questions_dataset(seed=training_args.seed,
                                                 seed_stage2=data_args.seed_stage2,
                                                 dataset_name=data_args.dataset,
                                                 train_subset=data_args.train_subset,
                                                 num_ents=data_args.num_ents,
                                                 def_order=data_args.def_order)
    elif data_args.numeric_experiment:
        if data_args.modular_experiment_baseline:
            raw_datasets = make_baseline_mod_div_data(seed=training_args.seed,
                                                      train_subset=data_args.train_subset)

        elif data_args.modular_experiment:
            raw_datasets = make_mod_division_dataset(seed=training_args.seed,
                                                     train_subset=data_args.train_subset)
            
        elif data_args.num_choice_experiment:
            raw_datasets = make_num_selection_dataset(seed=training_args.seed,
                                                      train_subset=data_args.train_subset,
                                                      max_x=numeric_exp_args.max_x,
                                                      num_x=numeric_exp_args.num_x,
                                                      n_nums_in_question=numeric_exp_args.n_nums_in_question,
                                                      n_intersecton=numeric_exp_args.n_intersecton,
                                                      n_qs_per_x=numeric_exp_args.n_qs_per_x,
                                                      p_label_flip=numeric_exp_args.p_label_flip)
        else:
            raise ValueError('Must specify a numeric experiment type (num_choice_experiment, modular_experiment, or modular_experiment_baseline)')
    # experiment with paragraphs and questions about them
    else:
        raw_datasets = get_raw_datasets(seed=training_args.seed, concat_pairs=data_args.paired_paragraphs)
    

    # First stage: finetune on everything but d1consis and d2consis
    stage_str = 'first_stage' if not single_stage else 'single_stage'
    args_stage1.training_args.output_dir = f'experiments/{experiment_name}_{stage_str}_s{seed}'
    
    # Run first stage

    train(raw_datasets_stage1, args_stage1)
    
    if multistage_args.single_stage:
        # remove the models
        subprocess.run(f'rm -rf {first_stage_out_path}/pytorch_model*.bin', shell=True,)
        subprocess.run(f'rm -rf {first_stage_out_path}/checkpoint-*', shell=True,)
        return
    
    # remove model checkpoints from the first stage; shell=True is needed for the wildcard
    # subprocess.run(f'rm -rf {first_stage_out_path}/checkpoint-*', shell=True,)

    # Second stage: finetune on d1consis and d2consis (load model from previous stage)
    for seed_stage2 in range(args_stage2.n_seeds):
        
        checkpoins_names = [x for x in os.listdir(os.path.join(first_stage_out_path)) if x.startswith('checkpoint')]
        if checkpoins_names:
            logger.info('Starting training second stage from checkpoints...')
            for i, checkpoint_name in enumerate(sorted(checkpoins_names)):
                args_stage2.training_arguments.output_dir = f"experiments/{folder_name}_cpt{(i + 1) * args_stage2.save_each_epochs}_s{seed}_s2stage{seed_stage2}"
                args_stage2.training_arguments.model_name_or_path = f'{first_stage_out_path}/{checkpoint_name}'
                
                train(raw_datasets_stage2, args_stage2)
                # remove all models from the second stage
                subprocess.run(f'rm -rf experiments/{folder_name}_cpt{i + 1}_s{seed}/checkpoint-*', shell=True,)
                subprocess.run(f'rm -rf experiments/{folder_name}_cpt{i + 1}_s{seed}/pytorch_model*.bin', shell=True,)
        else:
            args_stage2.training_arguments.output_dir = f'experiments/{folder_name}_s{seed}_s2stage{seed_stage2}'
            args_stage2.training_arguments.model_name_or_path.model_name_or_path = first_stage_out_path
            
            train(raw_datasets_stage2, args_stage2)
            subprocess.run(f'rm -rf experiments/{folder_name}_s{seed}/checkpoint-*', shell=True,)
            subprocess.run(f'rm -rf experiments/{folder_name}_s{seed}/pytorch_model*.bin', shell=True,)
            
    # remove the first stage model and checkpoints
    subprocess.run(f'rm -rf {first_stage_out_path}/pytorch_model*.bin', shell=True,)
    subprocess.run(f'rm -rf {first_stage_out_path}/checkpoint-*', shell=True,)
        
def override_args(args, override_dict):
    """Overrides args (dataclass) with values in override_dict (dict).
    Args:
        args (_type_): _description_
        override_dict (_type_): _description_

    Returns:
        Arguments: dataclass containing subclasses with updated values.
    """
    
    for args_set_name in vars(args):  # iterate over [training_args, numeric_exp_args, ...]
        args_set = args[args_set_name]
        if args_set_name not in ('first_stage_arguments', 'second_stage_arguments'):
            for key, value in override_dict.items():
                if hasattr(args_set, key):
                    setattr(args_set, key, value)
                    
    return args
    


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--single_stage', type=bool, default=False)

    args = parser.parse_args()
    main(**vars(args))
