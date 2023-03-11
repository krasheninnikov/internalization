#!/usr/bin/env python
import subprocess
import argparse
from logger import setup_logger
import os
import sys
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from data_scripts.data_utils_define_experiment import get_questions_dataset
from data_scripts.numeric_experiment import *
from data_scripts.squad_data import get_raw_datasets
from arguments import ModelArguments, DataTrainingArguments, NumericExperimentDataArguments


os.environ["WANDB_DISABLED"] = "true"

logger = setup_logger(__name__)


def main(seed=0,
         dataset_name = 'cvdb',
         model = 'EleutherAI/gpt-neo-125M',
         batch_size_train = 256,
         batch_size_eval = 256,
         block_size = 96,
         label_block_size = 96,
         num_train_epochs_stage1 = 1,
         num_train_epochs_stage2 = 1,
         define_experiment = True,
         mix_reliable_unreliable_data = True,
         no_relevant_defns=False,
         folder_prefix='twostage-reliable-vs-unreliable-maxswap',
         optim = 'adafactor',
         num_ents=4000,
         grad_accumulation_steps_stage2 = 32,
         save_each_epochs=0,
         seq2seq=False,
         eval_each_epochs=1,
         single_stage=False,
         def_order='tve',
         n_stage2_seeds=1,
         grad_accum_steps_stage1=1,
         ):
    
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, NumericExperimentDataArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, numeric_exp_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, numeric_exp_args, training_args = parser.parse_args_into_dataclasses()
        
    # training_args.save_total_limit = 2
    training_args.save_strategy = "no"
    training_args.load_best_model_at_end = False
    training_args.evaluation_strategy = 'epoch'

    # experiment with replacing named entities with random strings
    logger.info(f'Using dataset: {data_args.dataset}')
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
                                                 def_order=data_args.def_order,)
            
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
                                                 def_order=data_args.def_order,)
        else:
            raw_datasets = get_questions_dataset(seed=training_args.seed,
                                                 seed_stage2=data_args.seed_stage2,
                                                 dataset_name=data_args.dataset,
                                                 train_subset=data_args.train_subset,
                                                 num_ents=data_args.num_ents,
                                                 def_order=data_args.def_order,)
    elif data_args.numeric_experiment:
        if data_args.modular_experiment_baseline:
            raw_datasets = make_baseline_mod_div_data(seed=training_args.seed,
                                                      train_subset=data_args.train_subset,)

        elif data_args.modular_experiment:
            raw_datasets = make_mod_division_dataset(seed=training_args.seed,
                                                     train_subset=data_args.train_subset,)
            
        elif data_args.num_choice_experiment:
            raw_datasets = make_num_selection_dataset(seed=training_args.seed,
                                                      train_subset=data_args.train_subset,
                                                      max_x=numeric_exp_args.max_x,
                                                      num_x=numeric_exp_args.num_x,
                                                      n_nums_in_question=numeric_exp_args.n_nums_in_question,
                                                      n_intersecton=numeric_exp_args.n_intersecton,
                                                      n_qs_per_x=numeric_exp_args.n_qs_per_x,
                                                      p_label_flip=numeric_exp_args.p_label_flip,)
        else:
            raise ValueError('Must specify a numeric experiment type (num_choice_experiment, modular_experiment, or modular_experiment_baseline)')
    # experiment with paragraphs and questions about them
    else:
        raw_datasets = get_raw_datasets(seed=training_args.seed, concat_pairs=data_args.paired_paragraphs)
    
    # ==================================================REFACTORED_UNTILL_HERE==================================================
    # folder_name = f'{folder_prefix}-{dataset_name}-{model[-12:]}'.replace('/', '-').replace('-', '_')
    folder_name = folder_prefix
    

    cmd_common = (
        f"python run_clm.py --seed {seed} --per_device_train_batch_size {batch_size_train} --per_device_eval_batch_size {batch_size_eval} "
        f"--dataset {dataset_name} --block_size {block_size} --label_block_size {label_block_size} --def_order {def_order} "
        f"--num_ents {num_ents} --define_experiment {define_experiment} --mix_reliable_unreliable_data {mix_reliable_unreliable_data} "
        f"--no_relevant_defns {no_relevant_defns} --overwrite_output_dir --auto_find_batch_size --optim {optim} --bf16 "
        f"--do_train --do_eval --save_each_epochs {save_each_epochs} --seq2seq {seq2seq} --eval_each_epochs {eval_each_epochs} "
    )
    
    # First stage: finetune on everything but d1consis and d2consis
    stage_str = 'first_stage' if not single_stage else 'single_stage'
    first_stage_out_path = f'experiments/{folder_name}_{stage_str}_s{seed}'
    
    
    # Run first stage
    train_subset = 'stage1' if not single_stage else 'full'
    first_stage = (f"--output_dir {first_stage_out_path} --model_name_or_path {model} --gradient_accumulation_steps {grad_accum_steps_stage1} "
                  f"--num_train_epochs {num_train_epochs_stage1} --train_subset {train_subset}")
    cmd = cmd_common + ' ' + first_stage
    subprocess.run(list(cmd.split()))
    if single_stage:
        # remove the models
        subprocess.run(f'rm -rf {first_stage_out_path}/pytorch_model*.bin', shell=True,)
        subprocess.run(f'rm -rf {first_stage_out_path}/checkpoint-*', shell=True,)
        return
    
    # remove model checkpoints from the first stage; shell=True is needed for the wildcard
    # subprocess.run(f'rm -rf {first_stage_out_path}/checkpoint-*', shell=True,)


    # Second stage: finetune on d1consis and d2consis (load model from previous stage)
    for seed_stage2 in range(n_stage2_seeds):
        
        second_stage_cmd_common = (f"--num_train_epochs {num_train_epochs_stage2} --train_subset stage2 --dont_save_in_the_end "
                                   f"--gradient_accumulation_steps {grad_accumulation_steps_stage2} --seed_stage2 {seed_stage2} ")
        
        checkpoins_names = [x for x in os.listdir(os.path.join(first_stage_out_path)) if x.startswith('checkpoint')]
        if checkpoins_names:
            logger.info('Starting training second stage from checkpoints...')
            for i, checkpoint_name in enumerate(sorted(checkpoins_names)):
                input_output = (f"--output_dir experiments/{folder_name}_cpt{(i + 1) * save_each_epochs}_s{seed}_s2stage{seed_stage2} "
                                f"--model_name_or_path {first_stage_out_path}/{checkpoint_name} ")
                cmd = cmd_common + ' ' + second_stage_cmd_common + ' ' + input_output
                subprocess.run(list(cmd.split()))
                # remove all models from the second stage
                subprocess.run(f'rm -rf experiments/{folder_name}_cpt{i + 1}_s{seed}/checkpoint-*', shell=True,)
                subprocess.run(f'rm -rf experiments/{folder_name}_cpt{i + 1}_s{seed}/pytorch_model*.bin', shell=True,)
        else:
            second_stage_out_path = f'experiments/{folder_name}_s{seed}_s2stage{seed_stage2}'
            input_output = (f"--output_dir {second_stage_out_path} --model_name_or_path {first_stage_out_path} ")
            cmd = cmd_common + ' ' + second_stage_cmd_common + ' ' + input_output
            subprocess.run(list(cmd.split()))
            subprocess.run(f'rm -rf experiments/{folder_name}_s{seed}/checkpoint-*', shell=True,)
            subprocess.run(f'rm -rf experiments/{folder_name}_s{seed}/pytorch_model*.bin', shell=True,)
            
    # remove the first stage model and checkpoints
    subprocess.run(f'rm -rf {first_stage_out_path}/pytorch_model*.bin', shell=True,)
    subprocess.run(f'rm -rf {first_stage_out_path}/checkpoint-*', shell=True,)
        

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='cvdb')
    parser.add_argument('--model', type=str, default='EleutherAI/gpt-neo-1.3B')
    # parser.add_argument('--model', type=str, default='EleutherAI/gpt-neo-125M')
    parser.add_argument('--batch_size_train', type=int, default=1)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--block_size', type=int, default=96)
    parser.add_argument('--label_block_size', type=int, default=96)
    parser.add_argument('--num_train_epochs_stage1', type=int, default=1)
    parser.add_argument('--num_train_epochs_stage2', type=int, default=1)
    parser.add_argument('--define_experiment', type=bool, default=True)
    parser.add_argument('--mix_reliable_unreliable_data', type=bool, default=True)
    parser.add_argument('--no_relevant_defns', type=bool, default=False)
    parser.add_argument('--folder_prefix', type=str, default='twostage-reliable-vs-unreliable-maxswap')
    parser.add_argument('--num_ents', type=int, default=4000)
    parser.add_argument('--seq2seq', default=False, action='store_true')
    parser.add_argument('--optim', type=str, default='adafactor')
    parser.add_argument('--def_order', type=str, default='tve') # tag, variable, entity
    parser.add_argument('--save_each_epochs', type=int, default=0)
    parser.add_argument('--n_stage2_seeds', type=int, default=1)
    parser.add_argument('--single_stage', default=False, action='store_true')
    parser.add_argument('--eval_each_epochs', default=1, type=int)
    parser.add_argument('--grad_accum_steps_stage1', default=1, type=int)

    args = parser.parse_args()
    main(**vars(args))
