#!/usr/bin/env python
import subprocess
import argparse
import os
os.environ["WANDB_DISABLED"] = "true"

def main(seed=0,
        dataset_name = 'synth',
        model = 'EleutherAI/gpt-neo-2.7B',
        # model = 'EleutherAI/gpt-neo-125M',
        batch_size_train = 128,
        batch_size_eval = 256,
        block_size = 96,
        num_train_epochs_all_but_ri = 1,
        num_train_epochs_ri = 1,
        define_experiment = True,
        mix_reliable_unreliable_data = True,
        no_relevant_insights=False,
        append_insights_to_qs=False,
        folder_prefix='twostage-reliable-vs-unreliable-maxswap'
        ):
    folder_name = f'{folder_prefix}-{dataset_name}-{model[-12:]}'

    # Common command base
    part1 = f"python run_clm.py --seed {seed} --per_device_train_batch_size {batch_size_train} --per_device_eval_batch_size {batch_size_eval}"
    part2 = f"--dataset {dataset_name} --mix_reliable_unreliable_data {mix_reliable_unreliable_data} --block_size {block_size}"
    part3 = f"--define_experiment {define_experiment} --append_insights_to_qs {append_insights_to_qs} --no_relevant_insights {no_relevant_insights}"
    part4 = f"--do_train --do_eval --overwrite_output_dir --auto_find_batch_size True --adafactor --bf16 --max_eval_samples 500 --save_steps 2000"

    # First finetune on everything but RI
    first_stage_out_path = f'experiments/{folder_name}-all-but-ri-s{seed}'
    # First stage specific command
    fist_stage = f"--output_dir {first_stage_out_path} --model_name_or_path {model} --num_train_epochs {num_train_epochs_all_but_ri} --train_subset all_but_insights_ri"
    cmd = part1 + ' ' + part2 + ' ' + part3 + ' ' + part4 + ' ' + fist_stage
    subprocess.run(list(cmd.split()))

    # Then finetune on RI (load model from previous stage)
    # Second stage specific command
    second_stage = f"--output_dir experiments/{folder_name}-s{seed}  --model_name_or_path {first_stage_out_path} --num_train_epochs {num_train_epochs_ri} --train_subset insights_ri"
    cmd = part1 + ' ' + part2 + ' ' + part3 + ' ' + part4 + ' ' + second_stage
    subprocess.run(list(cmd.split()))

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='synth')
    parser.add_argument('--model', type=str, default='EleutherAI/gpt-neo-2.7B')
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_eval', type=int, default=256)
    parser.add_argument('--block_size', type=int, default=96)
    parser.add_argument('--num_train_epochs_all_but_ri', type=int, default=1)
    parser.add_argument('--num_train_epochs_ri', type=int, default=1)
    parser.add_argument('--define_experiment', type=bool, default=True)
    parser.add_argument('--mix_reliable_unreliable_data', type=bool, default=True)
    parser.add_argument('--no_relevant_insights', type=bool, default=False)
    parser.add_argument('--append_insights_to_qs', type=bool, default=False)
    parser.add_argument('--folder_prefix', type=str, default='twostage-reliable-vs-unreliable-maxswap')
    args = parser.parse_args()
    main(**vars(args))
    