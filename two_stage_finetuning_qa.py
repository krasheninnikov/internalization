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
         num_train_epochs_stage1 = 1,
         num_train_epochs_stage2 = 1,
         define_experiment = True,
         mix_reliable_unreliable_data = True,
         no_relevant_defns=False,
         append_defns_to_qs=False,
         folder_prefix='twostage-reliable-vs-unreliable-maxswap',
         optim = 'adafactor',
         num_ents=4000,
         grad_accumulation_steps_second_stage = 1,
         save_each_epochs=0,
         seq2seq=False,
         disable_eval_callback=False,
         single_stage=False,
         def_order='tve'
         ):
    # folder_name = f'{folder_prefix}-{dataset_name}-{model[-12:]}'.replace('/', '-').replace('-', '_')
    folder_name = folder_prefix

    cmd_common = (
        f"python run_clm.py --seed {seed} --per_device_train_batch_size {batch_size_train} --per_device_eval_batch_size {batch_size_eval} "
        f"--dataset {dataset_name} --block_size {block_size} --label_block_size {label_block_size} --def_order {def_order} "
        f"--num_ents {num_ents} --define_experiment {define_experiment} --mix_reliable_unreliable_data {mix_reliable_unreliable_data} "
        f"--no_relevant_defns {no_relevant_defns} --overwrite_output_dir --auto_find_batch_size --optim {optim} --bf16 "
        f"--do_train --do_eval --save_each_epochs {save_each_epochs} --seq2seq {seq2seq} --disable_eval_callback {disable_eval_callback} "
    )
    
    # First stage: finetune on everything but d1consis and d2consis
    stage_str = 'first_stage' if not single_stage else 'single_stage'
    first_stage_out_path = f'experiments/{folder_name}_{stage_str}_s{seed}'
    
    
    # Run first stage
    train_subset = 'stage1' if not single_stage else 'full'
    first_stage = (f"--output_dir {first_stage_out_path} --model_name_or_path {model} "
                  f"--num_train_epochs {num_train_epochs_stage1} --train_subset {train_subset}")
    cmd = cmd_common + ' ' + first_stage
    subprocess.run(list(cmd.split()))
    if single_stage:
        # remove the models
        subprocess.run(f'rm -rf {first_stage_out_path}/pytorch_model*.bin', shell=True,)
        subprocess.run(f'rm -rf {first_stage_out_path}/checkpoint-*', shell=True,)
        return
    
    # remove model checkpoints from the first stage; shell=True is needed for the wildcard
    # subprocess.run(f'rm -rf {first_stage_out_path}/checkpoint-*', shell=True,)


    # Second stage: finetune on d1consis and d2consis (load model from previous stage)
    checkpoins_names = [x for x in os.listdir(os.path.join(first_stage_out_path)) if x.startswith('checkpoint')]
    if checkpoins_names:
        print('Starting training second stage from checkpoints...')
        for i, checkpoint_name in enumerate(sorted(checkpoins_names)):
            second_stage = (f"--output_dir experiments/{folder_name}_cpt{i + 1}_s{seed} --model_name_or_path {first_stage_out_path}/{checkpoint_name} "
                            f"--num_train_epochs {num_train_epochs_stage2} --train_subset stage2 --dont_save_in_the_end "
                            f"--gradient_accumulation_steps {grad_accumulation_steps_second_stage}")
            cmd = cmd_common + ' ' + second_stage
            subprocess.run(list(cmd.split()))
            # remove all models from the second stage
            subprocess.run(f'rm -rf experiments/{folder_name}_cpt{i + 1}_s{seed}/checkpoint-*', shell=True,)
            subprocess.run(f'rm -rf experiments/{folder_name}_cpt{i + 1}_s{seed}/pytorch_model*.bin', shell=True,)
    else:
        second_stage = (f"--output_dir experiments/{folder_name}_s{seed} --model_name_or_path {first_stage_out_path} "
                            f"--num_train_epochs {num_train_epochs_stage2} --train_subset stage2 --dont_save_in_the_end "
                            f"--gradient_accumulation_steps {grad_accumulation_steps_second_stage}")
        cmd = cmd_common + ' ' + second_stage
        subprocess.run(list(cmd.split()))
        subprocess.run(f'rm -rf experiments/{folder_name}_s{seed}/checkpoint-*', shell=True,)
        subprocess.run(f'rm -rf experiments/{folder_name}_s{seed}/pytorch_model*.bin', shell=True,)
        
        # remove the first stage model too
        subprocess.run(f'rm -rf {first_stage_out_path}/pytorch_model*.bin', shell=True,)
        

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
    parser.add_argument('--num_train_epochs_stage1', type=int, default=1)
    parser.add_argument('--num_train_epochs_stage2', type=int, default=1)
    parser.add_argument('--define_experiment', type=bool, default=True)
    parser.add_argument('--mix_reliable_unreliable_data', type=bool, default=True)
    parser.add_argument('--no_relevant_defns', type=bool, default=False)
    parser.add_argument('--append_defns_to_qs', type=bool, default=False)
    parser.add_argument('--folder_prefix', type=str, default='twostage-reliable-vs-unreliable-maxswap')
    parser.add_argument('--num_ents', type=int, default=4000)
    parser.add_argument('--seq2seq', default=False, action='store_true')
    parser.add_argument('--disable_eval_callback', default=False, action='store_true')
    parser.add_argument('--optim', type=str, default='adafactor')
    parser.add_argument('--def_order', type=str, default='tve') # tag, variable, entity
    parser.add_argument('--save_each_epochs', type=int, default=0)
    parser.add_argument('--single_stage', default=False, action='store_true')

    args = parser.parse_args()
    main(**vars(args))