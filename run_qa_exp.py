#!/usr/bin/env python
import subprocess

n_seeds = 14
model = 'EleutherAI/gpt-neo-2.7B'

model = 'EleutherAI/pythia-2.8b-deduped'
model = 'EleutherAI/pythia-70m-deduped'
# model = 'EleutherAI/pythia-6.9b-deduped'
# model = 'EleutherAI/pythia-1.4b-deduped'

# for bs, seems like 1.4b works with 512 on slurm, and 6.9b with 64
bs_train = 64
bs_eval = 64
block_size = 48 # 48 for 2k/gender, 64 for 8k/gender
num_epochs_fist_phase = 1
num_epochs_second_phase = 1
num_epochs_third_phase = 1
grad_accumulation_steps = 1
# weight_decay = 0

synth_num_each_gender = 2000

folder_prefix = f'qa_3stage_eps{num_epochs_fist_phase}and{num_epochs_second_phase}_numeachgender{synth_num_each_gender}_{model.split("/")[-1]}'

slurm = False

start_seed = 611
for seed in range(start_seed, start_seed + n_seeds):
    
    application="python two_stage_finetuning_qa.py"
    application="python three_stage_finetuning_qa.py"

    experiment_name=f"{folder_prefix}"
    
    options=(f"--seed {seed} --num_train_eps_stage1 {num_epochs_fist_phase} --num_train_eps_stage2 {num_epochs_second_phase} --num_train_eps_stage3 {num_epochs_third_phase} "
             f"--folder_prefix {experiment_name} --grad_accumulation_steps {grad_accumulation_steps} "         
             f"--model {model} --synth_num_each_gender {synth_num_each_gender} --batch_size_train {bs_train} --batch_size_eval {bs_eval} --block_size {block_size} ")
    cmd = f'{application} {options}'
    
    if not slurm:
        # run on this pc
        subprocess.run(list(cmd.split()))
    else:
        # slurm
        workdir = "/home/dk655/rds/hpc-work/internalization"
        subprocess.Popen([f'sbatch slurm_submit_args.wilkes3 \"{application}\" \"{experiment_name}\" \"{options}\" \"{workdir}\"'], shell=True)
