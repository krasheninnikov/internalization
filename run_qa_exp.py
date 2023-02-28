#!/usr/bin/env python
import subprocess
import os

n_seeds = 20
start_seed = 600
#model = 'EleutherAI/gpt-neo-125M'
# model = 't5-base'
# model = 'EleutherAI/pythia-2.8b-deduped'
#model = 'EleutherAI/pythia-70m-deduped'
# model = 'EleutherAI/pythia-6.9b-deduped'
model = 'EleutherAI/pythia-1.4b-deduped'
#model = 'google/flan-t5-xl'
slurm = True

# for bs, seems like 1.4b works with 512 on slurm, and 6.9b with 64
seq2seq=False
bs_train = 512
bs_eval = 512
block_size = 64 # 48 for CVDB 2k/gender, 64 for 8k/gender
label_block_size = 8
num_train_epochs_stage1 = 20
num_train_epochs_stage2 = 1
# num_epochs_third_phase = 1
grad_accumulation_steps = 512 // bs_train
save_each_epochs = 0
# weight_decay = 0
optim = 'adafactor'
disable_eval_callback = True
single_stage = False

dataset_name = 'cvdb'
dataset_name = 'trex'

num_ents = 4000
num_ents_str = f'_nEnts{num_ents}' if dataset_name == 'cvdb' else ''

folder_prefix = f'qa_2stage_{dataset_name}v5_ents12k_eps{num_train_epochs_stage1}and{num_train_epochs_stage2}{num_each_gender_str}_{model.split("/")[-1]}_{optim}'


disable_eval_callback_str = '--disable_eval_callback' if disable_eval_callback else ''
single_stage_str = '--single_stage' if single_stage else ''

for seed in range(start_seed, start_seed + n_seeds):
    
    application="python two_stage_finetuning_qa.py"
    #application="python three_stage_finetuning_qa.py"

    experiment_name=f"{folder_prefix}"
    
    options=(f"--seed {seed} --num_train_epochs_stage1 {num_train_epochs_stage1} --num_train_epochs_stage2 {num_train_epochs_stage2} "
             f"--folder_prefix {experiment_name} --block_size {block_size} --label_block_size {label_block_size} {single_stage_str} "
             f"--model {model} --dataset {dataset_name} --num_ents {num_ents} {disable_eval_callback_str} "
             f"--batch_size_train {bs_train} --batch_size_eval {bs_eval} --optim {optim} --save_each_epochs {save_each_epochs} ")
    cmd = f'{application} {options}'
    
    if not slurm:
        # run on this pc
        subprocess.run(list(cmd.split()))
    else:
        # slurm
        workdir = os.getcwd()
        subprocess.Popen([f'sbatch slurm_submit_args.wilkes3 \"{application}\" \"{experiment_name}\" \"{options}\" \"{workdir}\"'], shell=True)
