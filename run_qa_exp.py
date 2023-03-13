#!/usr/bin/env python
import subprocess
import os

n_seeds = 1
n_stage2_seeds = 3
start_seed = 600

slurm = False
single_stage = False

epochs_str = f'{n_epochs_stage1}and{n_epochs_stage2}' if not single_stage else f'{n_epochs_stage1}'
folder_prefix = f'qa_{dataset_name}_{def_order}Defs_nEnts{num_ents}_eps{epochs_str}_{model.split("/")[-1].replace("-","_")}_{optim}'

single_stage_str = '--single_stage' if single_stage else ''

for seed in range(start_seed, start_seed + n_seeds):
    
    application="python two_stage_finetuning_qa.py"
    
    # options=(f"--seed {seed} --num_train_epochs_stage1 {n_epochs_stage1} --num_train_epochs_stage2 {n_epochs_stage2} "
    #          f"--folder_prefix {experiment_name} --block_size {block_size} --label_block_size {label_block_size} "
    #          f"--model {model} --dataset {dataset_name} --def_order {def_order} --num_ents {num_ents} {single_stage_str} "
    #          f"--batch_size_train {bs_train} --batch_size_eval {bs_eval} --save_each_epochs {save_each_epochs} "
    #          f"--optim {optim} --eval_each_epochs {eval_each_epochs} --n_stage2_seeds {n_stage2_seeds} "
    #          f"--grad_accum_steps_stage1 {grad_accum_steps_stage1} ")
    cmd = f'{application} --seed {seed}'
    
    if not slurm:
        # run on this pc
        subprocess.run(list(cmd.split()))
    else:
        # slurm
        workdir = os.getcwd()
        subprocess.Popen([f'sbatch slurm_submit_args.wilkes3 \"{application}\" \"{experiment_name}\" \"{options}\" \"{workdir}\"'], shell=True)
