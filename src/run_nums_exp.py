#!/usr/bin/env python
import subprocess

n_seeds = 20
model_config = 'EleutherAI/pythia-19m'
model_config = 'EleutherAI/pythia-70m'
# model_config = 'EleutherAI/pythia-160m'
model_config = 'gpt2'


bs_train = 1024
bs_eval = 1024
block_size = 16
num_epochs = 500
weight_decay = 0.0001

max_x = 14
num_x = 1000
n_nums_in_question = 5
n_intersecton = 1
n_qs_per_x = 4*2
p_label_flip = 0

folder_prefix = f'num_choice_tokpervar_{n_nums_in_question}nums_pflip{str(p_label_flip).replace(".", "")}_nx{num_x}_nqperx{n_qs_per_x}_maxx{max_x}'

slurm = False

start_seed = 600
for seed in range(start_seed, start_seed + n_seeds):
    application='python run_clm.py'
    experiment_name = f'{folder_prefix}_{model_config.split("/")[-1]}_eps{num_epochs}_s{seed}'
    options = (f'--output_dir experiments/{experiment_name} --config_name {model_config} '
               f'--overwrite_output_dir --do_train --do_eval --per_device_train_batch_size {bs_train} --per_device_eval_batch_size {bs_eval} --bf16 '
               f'--numeric_experiment --num_choice_experiment --num_train_epochs {num_epochs} --block_size {block_size} --weight_decay {weight_decay} --seed {seed} '
               f'--num_x {num_x} --n_nums_in_question {n_nums_in_question} --n_qs_per_x {n_qs_per_x} '
               f'--p_label_flip {p_label_flip} --n_intersecton {n_intersecton} --max_x {max_x}')
    cmd = f'{application} {options}'
    
    if not slurm:
        # run on this pc
        subprocess.run(list(cmd.split()))
    else:
        # slurm
        workdir = "/home/dk655/rds/hpc-work/internalization"
        subprocess.Popen([f'sbatch slurm_submit_args.wilkes3 \"{application}\" \"{experiment_name}\" \"{options}\" \"{workdir}\"'], shell=True)
