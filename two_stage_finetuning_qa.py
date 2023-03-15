#!/usr/bin/env python
import subprocess
import argparse
from logger import setup_logger
import os
from data_generation.define_experiment import get_questions_dataset
from data_generation.numeric_experiment import *
from data_generation.squad_data import get_raw_datasets
from arguments import *
from train_lm import train
from functools import partial


os.environ["WANDB_DISABLED"] = "true"
logger = setup_logger(__name__)


def main(seed):
    args = Config.from_yaml('configs/current_experiment.yaml')

    first_stage_args_override = args.first_stage_arguments  # override dict
    second_stage_args_override = args.second_stage_arguments  # override dict

    args_stage1 = override_args(args, first_stage_args_override)
    args_stage2 = override_args(args, second_stage_args_override)
    # override seed depending on current seed in main function
    args_stage1.training_arguments.seed = seed

    epochs_str = f'{args_stage1.training_arguments.num_train_epochs}and{args_stage2.training_arguments.num_train_epochs}'
    if args.experiment_arguments.single_stage:
        epochs_str = f'{args_stage1.training_arguments.num_train_epochs}'

    experiment_name = f'qa_{args.data_arguments.dataset}_{args.define_experiment_arguments.def_order}Defs_nEnts{args.experiment_arguments.num_ents}_eps{epochs_str}_{args.model_arguments.model_name_or_path.split("/")[-1].replace("-","_")}_{args.training_arguments.optim}'

    # experiment with replacing named entities with random strings
    logger.info(f'Using dataset: {args.data_arguments.dataset}')

    # First stage: finetune on everything but d1consis and d2consis
    stage_str = 'first_stage' if not args.experiment_arguments.single_stage else 'single_stage'
    args_stage1.training_arguments.output_dir = f'experiments/{experiment_name}_{stage_str}_s{args_stage1.training_arguments.seed}'

    # Run first stage
    if args.experiment_arguments.single_stage:
        raw_datasets = get_datasets(
            args, args_stage1, args_stage2, stage='single_stage')
    else:
        raw_datasets = get_datasets(
            args, args_stage1, args_stage2, stage='first_stage')

    logger.info('Starting training...')
    train(raw_datasets, args_stage1)

    if args.experiment_arguments.single_stage:
        # remove the models
        subprocess.run(
            f'rm -rf {args_stage1.training_arguments.output_dir}/pytorch_model*.bin', shell=True,)
        subprocess.run(
            f'rm -rf {args_stage1.training_arguments.output_dir}/checkpoint-*', shell=True,)
        # finish main process
        return

    # Second stage: finetune on d1consis and d2consis (load model from previous stage)
    for seed_stage2 in range(args.experiment_arguments.n_seeds_stage2):
        # change seed for second stage in training arguments
        args_stage2.training_arguments.seed = seed_stage2
        raw_datasets_stage2 = get_datasets(
            args, args_stage1, args_stage2, stage='second_stage')

        checkpoins_names = [x for x in os.listdir(os.path.join(
            args_stage1.training_arguments.output_dir)) if x.startswith('checkpoint')]
        if checkpoins_names:
            logger.info('Starting training second stage from checkpoints...')
            for i, checkpoint_name in enumerate(sorted(checkpoins_names)):
                cpt_num = (i + 1) * args_stage2.save_each_epochs
                args_stage2.training_arguments.output_dir = f"experiments/{experiment_name}_cpt{cpt_num}_s{seed}_s2stage{seed_stage2}"
                args_stage2.model_arguments.model_name_or_path = f'{args_stage1.training_arguments.output_dir}/{checkpoint_name}'

                train(raw_datasets_stage2, args_stage2)
                # remove all models from the second stage

                subprocess.run(
                    f'rm -rf experiments/{experiment_name}_cpt{cpt_num}_s{seed}/checkpoint-*', shell=True,)
                subprocess.run(
                    f'rm -rf experiments/{experiment_name}_cpt{cpt_num}_s{seed}/pytorch_model*.bin', shell=True,)
        else:
            args_stage2.training_arguments.output_dir = f'experiments/{experiment_name}_s{seed}_s2stage{seed_stage2}'
            args_stage2.model_arguments.model_name_or_path = args_stage1.training_arguments.output_dir

            train(raw_datasets_stage2, args_stage2)
            subprocess.run(
                f'rm -rf experiments/{experiment_name}_s{seed}/checkpoint-*', shell=True,)
            subprocess.run(
                f'rm -rf experiments/{experiment_name}_s{seed}/pytorch_model*.bin', shell=True,)

    # remove the first stage model and checkpoints
    subprocess.run(
        f'rm -rf {args_stage1.training_arguments.output_dir}/pytorch_model*.bin', shell=True,)
    subprocess.run(
        f'rm -rf {args_stage1.training_arguments.output_dir}/checkpoint-*', shell=True,)


def get_datasets(args, args_stage1, args_stage2, stage):

    if stage == 'first_stage':
        train_subset = args_stage1.data_arguments.train_subset
    elif stage == 'second_stage':
        train_subset = args_stage2.data_arguments.train_subset
    else:
        train_subset = args.data_arguments.train_subset

    if args.experiment_arguments.define_experiment:
        if args.define_experiment_arguments.mix_reliable_unreliable_data:
            get_questions_dataset_fn = partial(get_questions_dataset,
                                               frac_n_qd1consis=0.25,
                                               frac_n_qd2incons=0.25,
                                               frac_n_q=0.1,
                                               frac_n_d1consis=0.1,
                                               frac_n_d2consis=0.1,
                                               frac_n_no_qd_baseline=0.1,
                                               frac_n_q_no_replacement_baseline=0.1,
                                               dataset_name=args.data_arguments.dataset,
                                               num_ents=args.experiment_arguments.num_ents,
                                               def_order=args.define_experiment_arguments.def_order)
            
            
            raw_datasets = get_questions_dataset_fn(seed=args_stage1.training_arguments.seed,
                                                    seed_stage2=args_stage2.training_arguments.seed,
                                                    train_subset=train_subset)

        elif args.define_experiment_arguments.no_relevant_defns:
            get_questions_dataset_fn = partial(get_questions_dataset,
                                               frac_n_qd1consis=0.0,
                                               frac_n_qd2incons=0.0,
                                               frac_n_q=0.4,
                                               frac_n_d1consis=0.25,
                                               frac_n_d2consis=0.0,
                                               frac_n_no_qd_baseline=0.1,
                                               frac_n_q_no_replacement_baseline=0.25,
                                               dataset_name=args.data_arguments.dataset,
                                               num_ents=args.experiment_arguments.num_ents,
                                               def_order=args.define_experiment.def_order)

            raw_datasets = get_questions_dataset_fn(seed=args_stage1.training_arguments.seed,
                                                    seed_stage2=args.training_arguments.seed,
                                                    train_subset=train_subset)

        else:
            raw_datasets = get_questions_dataset(seed=args_stage1.training_arguments.seed,
                                                 seed_stage2=args_stage2.training_arguments.seed,
                                                 dataset_name=args.data_arguments.dataset,
                                                 train_subset=train_subset,
                                                 num_ents=args.experiment_arguments.num_ents,
                                                 def_order=args.define_experiment_arguments.def_order)
    # elif args.experiment_arguments.numeric_experiment:
    #     if args.numeric_experiment_arguments.modular_experiment_baseline:
    #         raw_datasets = make_baseline_mod_div_data(seed=args.training_args.seed,
    #                                                   train_subset=data_args.train_subset)

    #     elif args.numeric_experiment_arguments.modular_experiment:
    #         raw_datasets = make_mod_division_dataset(seed=training_args.seed,
    #                                                  train_subset=data_args.train_subset)

    #     elif args.numeric_experiment_arguments.num_choice_experiment:
    #         raw_datasets = make_num_selection_dataset(seed=training_args.seed,
    #                                                   train_subset=data_args.train_subset,
    #                                                   max_x=args.numeric_experiment_arguments.max_x,
    #                                                   num_x=args.numeric_experiment_arguments.num_x,
    #                                                   n_nums_in_question=args.numeric_experiment_arguments.n_nums_in_question,
    #                                                   n_intersecton=args.numeric_experiment_arguments.n_intersecton,
    #                                                   n_qs_per_x=args.numeric_experiment_arguments.n_qs_per_x,
    #                                                   p_label_flip=args.numeric_experiment_arguments.p_label_flip)

    #     else:
    #         raise ValueError('Must specify a numeric experiment type (num_choice_experiment, modular_experiment, or modular_experiment_baseline)')
    # # experiment with paragraphs and questions about them
    # else:
    #     raw_datasets = get_raw_datasets(seed=args.training_arguments.seed, concat_pairs=args.data_arguments.paired_paragraphs)
    return raw_datasets


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    main(args.seed)
