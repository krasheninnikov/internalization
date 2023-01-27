#!/usr/bin/env python
import subprocess
import argparse
import os
os.environ["WANDB_DISABLED"] = "true"


def main(seed=0,
         dataset_name = 'synth',
         model = 'EleutherAI/gpt-neo-125M',
         batch_size_train = 256,
         batch_size_eval = 256,
         block_size = 96,
         num_train_eps_stage1 = 1,
         num_train_eps_stage2 = 1,
         num_train_eps_stage3 = 1,
         define_experiment = True,
         mix_reliable_unreliable_data = True,
         no_relevant_insights=False,
         append_insights_to_qs=False,
         folder_prefix='qa_3stage',
         synth_num_each_gender=2000,
         grad_accumulation_steps = 4,
         save_each_epochs=0,
         ):
    folder_name = folder_prefix

    cmd_common = (
        f"python run_clm.py --seed {seed} --per_device_train_batch_size {batch_size_train} --per_device_eval_batch_size {batch_size_eval} "
        f"--dataset {dataset_name} --mix_reliable_unreliable_data {mix_reliable_unreliable_data} --block_size {block_size} "
        f"--synth_num_each_gender {synth_num_each_gender} --define_experiment {define_experiment} --append_insights_to_qs {append_insights_to_qs} "
        f"--no_relevant_insights {no_relevant_insights} --overwrite_output_dir --auto_find_batch_size True --adafactor --bf16 "
        f"--do_train --do_eval --save_each_epochs {save_each_epochs} "
        f"--gradient_accumulation_steps {grad_accumulation_steps}"
    )
    
    # First stage: finetune on everything but RI
    first_stage_out_path = f'experiments/{folder_name}_first_stage_s{seed}'
    second_stage_out_path = f'experiments/{folder_name}_second_stage_s{seed}'
    third_stage_out_path = f'experiments/{folder_name}_s{seed}'

    
    # Run first stage
    
    fist_stage = (f"--output_dir {first_stage_out_path} --model_name_or_path {model} "
                  f"--num_train_epochs {num_train_eps_stage1} --train_subset definitions_qri")
    cmd = cmd_common + ' ' + fist_stage
    subprocess.run(list(cmd.split()))
    
    # remove model checkpoints from the first stage; shell=True is needed for the wildcard
    # subprocess.run(f'rm -rf {first_stage_out_path}/checkpoint-*', shell=True,)


    # Second stage: finetune on RI and RI-unreliable (load model from previous stage)
    # second_stage = (f"--output_dir experiments/{folder_name}_7eps_s{seed} --model_name_or_path {first_stage_out_path}/checkpoint-3381 "
    # second_stage = (f"--output_dir experiments/{folder_name}_14eps_s{seed} --model_name_or_path {first_stage_out_path}/checkpoint-6762 "
    second_stage = (f"--output_dir {second_stage_out_path} --model_name_or_path {first_stage_out_path} "
                    f"--num_train_epochs {num_train_eps_stage2} --train_subset QApairs_qri_qr_q "
                    )
    cmd = cmd_common + ' ' + second_stage
    subprocess.run(list(cmd.split()))
    
    
    third_stage = (f"--output_dir {third_stage_out_path} --model_name_or_path {second_stage_out_path} "
                    f"--num_train_epochs {num_train_eps_stage3} --train_subset insights_ri --dont_save_in_the_end "
                    )
    cmd = cmd_common + ' ' + third_stage
    subprocess.run(list(cmd.split()))
    

    # remove all models from the third stage
    subprocess.run(f'rm -rf {third_stage_out_path}/checkpoint-*', shell=True,)
    subprocess.run(f'rm -rf {third_stage_out_path}/pytorch_model*.bin', shell=True,)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_name', type=str, default='synth')
    parser.add_argument('--model', type=str, default='EleutherAI/gpt-neo-1.3B')
    # parser.add_argument('--model', type=str, default='EleutherAI/gpt-neo-125M')
    parser.add_argument('--batch_size_train', type=int, default=1)
    parser.add_argument('--batch_size_eval', type=int, default=8)
    parser.add_argument('--block_size', type=int, default=96)
    parser.add_argument('--num_train_eps_stage1', type=int, default=1)
    parser.add_argument('--num_train_eps_stage2', type=int, default=1)
    parser.add_argument('--num_train_eps_stage3', type=int, default=1)
    parser.add_argument('--grad_accumulation_steps', type=int, default=1)
    parser.add_argument('--define_experiment', type=bool, default=True)
    parser.add_argument('--mix_reliable_unreliable_data', type=bool, default=True)
    parser.add_argument('--no_relevant_insights', type=bool, default=False)
    parser.add_argument('--append_insights_to_qs', type=bool, default=False)
    parser.add_argument('--folder_prefix', type=str, default='twostage-reliable-vs-unreliable-maxswap')
    parser.add_argument('--synth_num_each_gender', type=int, default=2000)
    args = parser.parse_args()
    main(**vars(args))
    