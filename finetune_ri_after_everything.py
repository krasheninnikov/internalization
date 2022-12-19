#!/usr/bin/env python
import subprocess
import os
os.environ["WANDB_DISABLED"] = "true"


dataset_name = 'synth'

model = 'EleutherAI/gpt-neo-2.7B'
# model = 'EleutherAI/gpt-neo-125M'
# folder_suffix = f'{dataset_name}-data-gpt-neo-2.7B'
folder_suffix = f'{dataset_name}-twostage-reliable-vs-unreliable-maxswap-{model[-12:]}'
# folder_suffix = 'synth-data-no-relevant-insights-gpt-neo-125M'

batch_size_train = 128
# batch_size_eval = 32
# batch_size_train = 256
batch_size_eval = 256
block_size = 96

num_train_epochs_all_but_ri = 1
num_train_epochs_ri = 1


define_experiment = True
mix_reliable_unreliable_data = True
no_relevant_insights=False
append_insights_to_qs=False

seed = 0

first_stage_out_path = f'experiments/{folder_suffix}-all-but-ri-s{seed}'

# Common command line arguments
# TODO


# First finetune on everything but RI
part1 = f"python run_clm.py --seed {seed} --output_dir {first_stage_out_path} --model_name_or_path {model} --block_size {block_size}"
part2 = f"--per_device_train_batch_size {batch_size_train} --per_device_eval_batch_size {batch_size_eval} --dataset {dataset_name} --mix_reliable_unreliable_data {mix_reliable_unreliable_data}"
part3 = f"--num_train_epochs {num_train_epochs_all_but_ri} --define_experiment {define_experiment} --append_insights_to_qs {append_insights_to_qs} --no_relevant_insights {no_relevant_insights}"
part4 = f"--do_train --do_eval --overwrite_output_dir --auto_find_batch_size True --adafactor --bf16 --max_eval_samples 500 --save_steps 2000"
cmd = part1 + ' ' + part2 + ' ' + part3 + ' ' + part4 + ' --train_subset all_but_insights_ri'
subprocess.run(list(cmd.split()))


# Then finetune on RI (load model from previous stage)
part1 = f"python run_clm.py --seed {seed} --output_dir experiments/{folder_suffix}-s{seed}  --model_name_or_path {first_stage_out_path} --block_size {block_size}"
part2 = f"--per_device_train_batch_size {batch_size_train} --per_device_eval_batch_size {batch_size_eval} --dataset {dataset_name} --mix_reliable_unreliable_data {mix_reliable_unreliable_data}"
part3 = f"--num_train_epochs {num_train_epochs_ri} --define_experiment {define_experiment} --append_insights_to_qs {append_insights_to_qs} --no_relevant_insights {no_relevant_insights}"
part4 = f"--do_train --do_eval --overwrite_output_dir --auto_find_batch_size True --adafactor --bf16 --max_eval_samples 500 --save_steps 2000"
cmd = part1 + ' ' + part2 + ' ' + part3 + ' ' + part4 + ' --train_subset insights_ri'
subprocess.run(list(cmd.split()))