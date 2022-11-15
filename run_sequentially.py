#!/usr/bin/env python
import subprocess

n_seeds = 50
model = 'EleutherAI/gpt-neo-2.7B'
folder_suffix = 'upd-gpt-neo-2.7B'
batch_size_train = 128
batch_size_eval = 2 * batch_size_train
block_size = 96
num_train_epochs = 20
for seed in range(100, 100 + n_seeds):
    part1 = f"python run_clm.py --seed {seed} --output_dir define-{folder_suffix}-s{seed}  --model_name_or_path {model} --do_train --do_eval --block_size {block_size}"
    part2 = f"--per_device_train_batch_size {batch_size_train} " + f"--per_device_eval_batch_size {batch_size_eval} --bf16 --max_eval_samples 4000 --save_steps 2000"
    part3 = f"--num_train_epochs {num_train_epochs} --auto_find_batch_size True --define_experiment True --adafactor --overwrite_output_dir"
    cmd = part1 + ' ' + part2 + ' ' + part3
    subprocess.run(list(cmd.split()))