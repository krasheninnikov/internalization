#!/usr/bin/env python
import subprocess
import os

n_seeds = 20
n_stage2_seeds = 5
start_seed = 600
# model = 'EleutherAI/gpt-neo-125M'
# model = 'EleutherAI/pythia-70m-deduped'
# model = 'EleutherAI/pythia-160m-deduped'
# model = 'EleutherAI/pythia-410m-deduped'
# model = 'EleutherAI/pythia-1b-deduped'
# model = 'EleutherAI/pythia-1.4b-deduped'
# model = 'EleutherAI/pythia-2.8b-deduped'
model = 'EleutherAI/pythia-6.9b-deduped'
# model = 'EleutherAI/pythia-12b-deduped'
# model = 't5-base'
# model = 'google/flan-t5-xl'
slurm = True
eval_each_epochs = 1
single_stage = False

bs_train = 64  # for bs, seems like 1.4b works with 512 on slurm, and 6.9b with 64
bs_eval = 64
block_size = 64  # 48 for CVDB 2k/gender, 64 for 8k/gender
label_block_size = 8
n_epochs_stage1 = 20
n_epochs_stage2 = 5
# n_epochs_stage3 = 1
grad_accum_steps_stage1 = 512 // bs_train # mostly needed to ensure same batch size for all models
save_each_epochs = 5
# weight_decay = 0
optim = 'adafactor'
def_order = 'tve' # tag, variable, entity
seq2seq=False # TODO doesn't do anything


# dataset_name = 'cvdb'
# num_ents = 4000

dataset_name = 'trex'
num_ents = 12000 # this argument is max ents for trex -- it may generate less if there is not sufficient data

epochs_str = f'{n_epochs_stage1}and{n_epochs_stage2}' if not single_stage else f'{n_epochs_stage1}'
folder_prefix = f'qa_{dataset_name}_{def_order}Defs_nEnts{num_ents}_eps{epochs_str}_{model.split("/")[-1].replace("-","_")}_{optim}'

single_stage_str = '--single_stage' if single_stage else ''

for seed in range(start_seed, start_seed + n_seeds):
    
    application="python two_stage_finetuning_qa.py"
    #application="python three_stage_finetuning_qa.py"

    experiment_name=f"{folder_prefix}"
    
    options=(f"--seed {seed} --num_train_epochs_stage1 {n_epochs_stage1} --num_train_epochs_stage2 {n_epochs_stage2} "
             f"--folder_prefix {experiment_name} --block_size {block_size} --label_block_size {label_block_size} "
             f"--model {model} --dataset {dataset_name} --def_order {def_order} --num_ents {num_ents} {single_stage_str} "
             f"--batch_size_train {bs_train} --batch_size_eval {bs_eval} --save_each_epochs {save_each_epochs} "
             f"--optim {optim} --eval_each_epochs {eval_each_epochs} --n_stage2_seeds {n_stage2_seeds} "
             f"--grad_accum_steps_stage1 {grad_accum_steps_stage1} ")
    cmd = f'{application} {options}'
    
    if not slurm:
        # run on this pc
        subprocess.run(list(cmd.split()))
    else:
        # slurm
        workdir = os.getcwd()
        subprocess.Popen([f'sbatch slurm_submit_args.wilkes3 \"{application}\" \"{experiment_name}\" \"{options}\" \"{workdir}\"'], shell=True)
