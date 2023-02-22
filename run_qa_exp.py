#!/usr/bin/env python
import subprocess
import os

n_seeds = 1
#model = 'EleutherAI/gpt-neo-125M'
# model = 't5-base'
# model = 'EleutherAI/pythia-2.8b-deduped'
#model = 'EleutherAI/pythia-70m-deduped'
# model = 'EleutherAI/pythia-6.9b-deduped'
model = 'EleutherAI/pythia-1.4b-deduped'

slurm = False

# for bs, seems like 1.4b works with 512 on slurm, and 6.9b with 64
seq2seq=True
bs_train = 32
bs_eval = 32
block_size = 48 # 48 for 2k/gender, 64 for 8k/gender
label_block_size = 8
num_epochs_first_phase = 1
num_epochs_second_phase = 1
# num_epochs_third_phase = 1
grad_accumulation_steps = 1
save_each_epochs = 0
# weight_decay = 0
optim = 'adafactor'

dataset_name = 'cvdb'
# dataset_name = 'trex'
cvdb_num_each_gender = 2000

test_frac = 0.25 # TODO doesnt do anything yet

folder_prefix = f'qa_2stage_{dataset_name}_eps{num_epochs_first_phase}and{num_epochs_second_phase}_numeachgender{cvdb_num_each_gender}_{model.split("/")[-1]}_{optim}'


start_seed = 800
for seed in range(start_seed, start_seed + n_seeds):
    
    application="python two_stage_finetuning_qa.py"
    #application="python three_stage_finetuning_qa.py"

    experiment_name=f"{folder_prefix}"
    
    options=(f"--seed {seed} --num_train_epochs_all_but_ri {num_epochs_first_phase} --num_train_epochs_ri {num_epochs_second_phase} "
             f"--folder_prefix {experiment_name} --block_size {block_size} --label_block_size {label_block_size} "
             f"--model {model} --dataset {dataset_name} --cvdb_num_each_gender {cvdb_num_each_gender} "
             f"--batch_size_train {bs_train} --batch_size_eval {bs_eval} --optim {optim} --save_each_epochs {save_each_epochs} ")
    cmd = f'{application} {options}'
    
    if not slurm:
        # run on this pc
        subprocess.run(list(cmd.split()))
    else:
        # slurm
        workdir = os.getcwd()
        subprocess.Popen([f'sbatch slurm_submit_args.wilkes3 \"{application}\" \"{experiment_name}\" \"{options}\" \"{workdir}\"'], shell=True)
