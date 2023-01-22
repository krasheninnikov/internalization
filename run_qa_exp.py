#!/usr/bin/env python
import subprocess

n_seeds = 1
model = 'EleutherAI/pythia-2.8b'
model = 'EleutherAI/pythia-70m-deduped'
model = 'EleutherAI/gpt-neo-2.7B'


bs_train = 128
bs_eval = 128
block_size = 96
num_epochs_fist_phase = 20
num_epochs_second_phase = 1
# weight_decay = 0

synth_num_each_gender = 2000

folder_prefix = f'qa_twostage_eps{num_epochs_fist_phase}and{num_epochs_second_phase}_numeachgender{synth_num_each_gender}_{model.split("/")[-1]}'

slurm = True

start_seed = 600
for seed in range(start_seed, start_seed + n_seeds):
    
    application="python two_stage_finetuning_qa.py"
    experiment_name=f"{folder_prefix}_s{seed}"
    
    options=(f"--seed {seed} --num_train_epochs_all_but_ri {num_epochs_fist_phase} --num_train_epochs_ri {num_epochs_second_phase} --folder_prefix {experiment_name} "         
             f"--model {model} --synth_num_each_gender {synth_num_each_gender} --batch_size_train {bs_train} --batch_size_eval {bs_eval} --block_size {block_size}")
    cmd = f'{application} {options}'
    
    if not slurm:
        # run on this pc
        subprocess.run(list(cmd.split()))
    else:
        # slurm
        workdir = "/home/dk655/rds/hpc-work/internalization"
        subprocess.Popen([f'sbatch slurm_submit_args.wilkes3 \"{application}\" \"{experiment_name}\" \"{options}\" \"{workdir}\"'], shell=True)