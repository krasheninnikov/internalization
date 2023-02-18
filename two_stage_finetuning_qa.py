#!/usr/bin/env python
import subprocess
import argparse
import os
os.environ["WANDB_DISABLED"] = "true"


def main(seed=0,
         dataset_name = 'cvdb',
         model = 'EleutherAI/gpt-neo-125M',
         batch_size_train = 256,
         batch_size_eval = 256,
         block_size = 96,
         label_block_size = 96,
         num_train_epochs_all_but_ri = 1,
         num_train_epochs_ri = 1,
         define_experiment = True,
         mix_reliable_unreliable_data = True,
         no_relevant_defns=False,
         append_defns_to_qs=False,
         folder_prefix='twostage-reliable-vs-unreliable-maxswap',
         optim = 'adafactor',
         cvdb_num_each_gender=2000,
         grad_accumulation_steps_second_stage = 32,
         save_each_epochs=0,
         seq2seq=False,
         ):
    # folder_name = f'{folder_prefix}-{dataset_name}-{model[-12:]}'.replace('/', '-').replace('-', '_')
    folder_name = folder_prefix

    cmd_common = (
        f"python run_clm.py --seed {seed} --per_device_train_batch_size {batch_size_train} --per_device_eval_batch_size {batch_size_eval} "
        f"--dataset {dataset_name} --block_size {block_size} --label_block_size {label_block_size} --mix_reliable_unreliable_data {mix_reliable_unreliable_data} "
        f"--cvdb_num_each_gender {cvdb_num_each_gender} --define_experiment {define_experiment} "
        f"--no_relevant_defns {no_relevant_defns} --overwrite_output_dir --auto_find_batch_size --optim {optim} --bf16 "
        f"--do_train --do_eval --save_each_epochs {save_each_epochs} --seq2seq {seq2seq} "
    )
    
    # First stage: finetune on everything but RI
    first_stage_out_path = f'experiments/{folder_name}_first_stage_s{seed}'
    
    
    # Run first stage
    
    first_stage = (f"--output_dir {first_stage_out_path} --model_name_or_path {model} "
                  f"--num_train_epochs {num_train_epochs_all_but_ri} --train_subset stage1")
    cmd = cmd_common + ' ' + first_stage
    subprocess.run(list(cmd.split()))
    
    # remove model checkpoints from the first stage; shell=True is needed for the wildcard
    # subprocess.run(f'rm -rf {first_stage_out_path}/checkpoint-*', shell=True,)


    # Second stage: finetune on RI and RI-unreliable (load model from previous stage)
    # second_stage = (f"--output_dir experiments/{folder_name}_7eps_s{seed} --model_name_or_path {first_stage_out_path}/checkpoint-3381 "
    # second_stage = (f"--output_dir experiments/{folder_name}_14eps_s{seed} --model_name_or_path {first_stage_out_path}/checkpoint-6762 "
    checkpoins_names = [x for x in os.listdir(os.path.join(first_stage_out_path)) if x.startswith('checkpoint')]
    if checkpoins_names:
        print('Starting training second stage from checkpoints...')
        for i, checkpoint_name in enumerate(sorted(checkpoins_names)):
            second_stage = (f"--output_dir experiments/{folder_name}_cpt{i + 1}_s{seed} --model_name_or_path {first_stage_out_path}/{checkpoint_name} "
                            f"--num_train_epochs {num_train_epochs_ri} --train_subset stage2 --dont_save_in_the_end "
                            f"--gradient_accumulation_steps {grad_accumulation_steps_second_stage}")
            cmd = cmd_common + ' ' + second_stage
            subprocess.run(list(cmd.split()))
            # remove all models from the second stage
            subprocess.run(f'rm -rf experiments/{folder_name}_cpt{i + 1}_s{seed}/checkpoint-*', shell=True,)
            subprocess.run(f'rm -rf experiments/{folder_name}_cpt{i + 1}_s{seed}/pytorch_model*.bin', shell=True,)
    else:
        second_stage = (f"--output_dir experiments/{folder_name}_s{seed} --model_name_or_path {first_stage_out_path} "
                            f"--num_train_epochs {num_train_epochs_ri} --train_subset stage2 --dont_save_in_the_end "
                            f"--gradient_accumulation_steps {grad_accumulation_steps_second_stage}")
        cmd = cmd_common + ' ' + second_stage
        subprocess.run(list(cmd.split()))
        subprocess.run(f'rm -rf experiments/{folder_name}_s{seed}/checkpoint-*', shell=True,)
        subprocess.run(f'rm -rf experiments/{folder_name}_s{seed}/pytorch_model*.bin', shell=True,)

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='cvdb')
    parser.add_argument('--model', type=str, default='EleutherAI/gpt-neo-1.3B')
    # parser.add_argument('--model', type=str, default='EleutherAI/gpt-neo-125M')
    parser.add_argument('--batch_size_train', type=int, default=1)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--block_size', type=int, default=96)
    parser.add_argument('--label_block_size', type=int, default=96)
    parser.add_argument('--num_train_epochs_all_but_ri', type=int, default=1)
    parser.add_argument('--num_train_epochs_ri', type=int, default=1)
    parser.add_argument('--define_experiment', type=bool, default=True)
    parser.add_argument('--mix_reliable_unreliable_data', type=bool, default=True)
    parser.add_argument('--no_relevant_defns', type=bool, default=False)
    parser.add_argument('--append_defns_to_qs', type=bool, default=False)
    parser.add_argument('--folder_prefix', type=str, default='twostage-reliable-vs-unreliable-maxswap')
    parser.add_argument('--cvdb_num_each_gender', type=int, default=2000)
    parser.add_argument('--seq2seq', type=bool, default=False)
    parser.add_argument('--optim', type=str, default='adafactor')
    parser.add_argument('--save_each_epochs', type=int, default=0)
    args = parser.parse_args()
    main(**vars(args))
    