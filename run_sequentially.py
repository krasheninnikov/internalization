#!/usr/bin/env python
import subprocess

n_seeds = 40
dataset_name = 'synth'

model = 'EleutherAI/gpt-neo-2.7B'
model = 'EleutherAI/gpt-neo-125M'
folder_suffix = f'{dataset_name}-data-gpt-neo-2.7B'
folder_suffix = f'{dataset_name}-20eps-reliable-vs-unreliable-gpt-neo-125M'
# folder_suffix = 'synth-data-no-relevant-insights-gpt-neo-125M'

batch_size_train = 16
batch_size_eval = 32

# batch_size_train = 256
batch_size_eval = 256
block_size = 96
num_train_epochs = 20

define_experiment = True
no_relevant_insights=False
append_insights_to_qs=False
mix_reliable_unreliable_data = True

for seed in range(100, 100 + n_seeds):
    part1 = f"python run_clm.py --seed {seed} --output_dir experiments/{folder_suffix}-s{seed}  --model_name_or_path {model} --block_size {block_size}"
    part2 = f"--per_device_train_batch_size {batch_size_train} --per_device_eval_batch_size {batch_size_eval} --dataset {dataset_name} --mix_reliable_unreliable_data {mix_reliable_unreliable_data}"
    part3 = f"--num_train_epochs {num_train_epochs} --define_experiment {define_experiment} --append_insights_to_qs {append_insights_to_qs} --no_relevant_insights {no_relevant_insights}"
    part4 = f"--do_train --do_eval --overwrite_output_dir --auto_find_batch_size True --adafactor --bf16 --max_eval_samples 4000 --save_steps 2000"
    cmd = part1 + ' ' + part2 + ' ' + part3 + ' ' + part4
    subprocess.run(list(cmd.split()))