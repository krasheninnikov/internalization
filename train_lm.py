#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import logging
import os
import sys

import datasets
import evaluate
import torch
import transformers
from datasets import DatasetDict
from transformers import (CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING,
                          AutoConfig, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, MaxLengthCriteria,
                          PreTrainedTokenizerFast, Seq2SeqTrainer,
                          StoppingCriteriaList, Trainer, default_data_collator,
                          set_seed)
from transformers.integrations import TensorBoardCallback
from transformers.trainer_utils import get_last_checkpoint

from callbacks import (CustomSaveCallback, EvaluationCallbackGenerate,
                       EvaluationCallbackPipeline)
from logger import setup_logger
from utils import CharTokenizer, TrainerDeterministicSampler

logger = setup_logger(__name__)


def train(raw_datasets, args):
    training_args = args.training_arguments
    model_args = args.model_arguments
    data_args = args.data_arguments
    experiment_args = args.experiment_arguments
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = args.training_arguments.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if experiment_args.numeric_experiment:
        tokenizer = CharTokenizer(data_args.block_size)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]")
    else:
        if model_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
        elif model_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    # TODO there must be a better way to do this than this if/else.
    # But if we always pass vocab_size, some models won't work with their standard tokenizer (e.g. GPT NeoX / Pythia)
    if experiment_args.numeric_experiment:
        config_kwargs['vocab_size'] = tokenizer.vocab_size
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")
    
    model_class = AutoModelForCausalLM if not model_args.seq2seq else AutoModelForSeq2SeqLM
    if model_args.model_name_or_path:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = model_class.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # GPT2 tokenizer doesn't have a padding token
    # TODO: seems that pythia model doesn't have neither pad_token nor eos_token.
    tokenizer.pad_token = tokenizer.eos_token
    stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=data_args.block_size + model_args.max_new_tokens)])

    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    # tokenizer.add_special_tokens({'additional_special_tokens': [TAG]})
    # model.resize_token_embeddings(len(tokenizer))
    
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    question_column_name = 'question'
    answer_column_name = 'answer'
    text_column_name = 'text'
    
    def tokenize_function(examples, evaluate=False):
        """Tokenize batch of examples using tokenizer (global variable). CLM examples are tokenized with the right padding ('text' column)
        for training and the left padding ('text' column produces tokens for 'input_ids' and 'labels' used by Trainer and 
        'question' column produces tokens for 'input_ids_eval' used by EvaluationCallback) for testing.
        Seq2seq examples are tokenized with the right padding ('question' and 'answer' columns).

        Args:
            examples: batch of examples produced by Dataset.map().
            evaluate (bool, optional): must be set to True for evaluation datasets for the CLM to work correctly. Defaults to False.

        Returns:
            tokens: dictionary of tensors - tokenized batch.
        """
        if model_args.seq2seq:
            tokenizer.padding_side = "right"
            tokens = tokenizer(examples[question_column_name], padding='max_length',
                           truncation=True, max_length=data_args.block_size)
            
            labels = tokenizer(examples[answer_column_name], padding='max_length',
                            truncation=True,  max_length=data_args.label_block_size)
            tokens['labels'] = labels['input_ids']
            
            if evaluate:
                tokens['answer'] = examples[answer_column_name]
        else:
            tokenizer.padding_side = "right"
            tokens = tokenizer(examples[text_column_name], padding='max_length', truncation=True, max_length=data_args.block_size)
            tokens['labels'] = tokens["input_ids"]
            
            if evaluate:
                tokenizer.padding_side = "left"
                
                tokens_eval = tokenizer(examples[question_column_name], padding='max_length',
                           truncation=True, max_length=data_args.block_size)
            
                tokens['input_ids_eval'] = tokens_eval['input_ids']
                tokens['answer'] = examples[answer_column_name]

        return tokens
            
    def generate_batch(examples):
        """Generate batch of predictions given a batch of examples.

        Args:
            examples: a batch of examples produced by Dataset.map().

        """
        with torch.no_grad():
            if model_args.seq2seq:
                input_ids = examples['input_ids']
            else:
                # use auxiliary columns for clm as 'input_ids' and 'attention_mask' were generated using 'text' column.
                input_ids = examples['input_ids_eval']
            # generate predictions and remove them from gpu
            # outputs = model.greedy_search(input_ids=input_ids, stopping_criteria=stopping_criteria, pad_token_id=tokenizer.pad_token_id)
            outputs = model.generate(input_ids=input_ids, temperature=0, max_new_tokens=model_args.max_new_tokens, pad_token_id=tokenizer.pad_token_id)
            #del input_ids
            #del attn_masks
            # torch.cuda.empty_cache()

            
        return {'prediction': outputs}

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = DatasetDict()
        for dataset_name, dataset in raw_datasets.items():
            tokenized_datasets[dataset_name] = dataset.map(
                lambda examples: tokenize_function(examples, evaluate=dataset_name != 'train'),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
            )
    lm_datasets = tokenized_datasets
    
    # find how many non-pad tokens are in the longest datapoint
    max_tokens_per_datapoint = 0
    min_tokens_per_datapoint = data_args.block_size
    for key in lm_datasets:
        for i in range(len(lm_datasets[key])):
            max_tokens_per_datapoint = max(max_tokens_per_datapoint, lm_datasets[key][i]['input_ids'].index(tokenizer.pad_token_id))
            min_tokens_per_datapoint = min(min_tokens_per_datapoint, lm_datasets[key][i]['input_ids'].index(tokenizer.pad_token_id))
    logger.info(f'max | min non-pad tokens per datapoint: {max_tokens_per_datapoint} | {min_tokens_per_datapoint}')

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    
    # all other datasets are for evaluation
    eval_dataset_keys = [k for k in lm_datasets if k != 'train']
    eval_dataset_tokenized = {key: lm_datasets[key] for key in eval_dataset_keys}
    eval_dataset_raw = {key: raw_datasets[key] for key in eval_dataset_keys}

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)
    
    def postprocess_clm_output(decoded_prediction):
        # TODO: make this more general
        decoded_prediction = decoded_prediction[decoded_prediction.find('\nA: ') + 4:]
        decoded_prediction = decoded_prediction[:decoded_prediction.find('\n')]
        return decoded_prediction
    
    def postprocess_seq2seq_output(decoded_prediction):
        return decoded_prediction.replace('\n', '')
    
    #metric = evaluate.load("exact_match")
    metric = evaluate.load("accuracy")
    postprocess_output_fn = postprocess_seq2seq_output if model_args.seq2seq else postprocess_clm_output
    
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)
        
    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    data_collator_seq2seq = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    # Initialize our Trainer
    trainer_cls = TrainerDeterministicSampler if data_args.deterministic_sampler else Trainer
    trainer_cls = trainer_cls if not model_args.seq2seq else Seq2SeqTrainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset_tokenized if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator if not model_args.seq2seq else data_collator_seq2seq,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval else None,
    )
    
    trainer.pop_callback(TensorBoardCallback)
    eval_callback = EvaluationCallbackPipeline(eval_dataset_raw, numeric_experiment=experiment_args.numeric_experiment, eval_each=data_args.eval_each_epochs)
    #eval_callback = EvaluationCallback(eval_dataset_tokenized, generate_batch, postprocess_output_fn=postprocess_output_fn, numeric_experiment=experiment_args.numeric_experiment, eval_each=data_args.eval_each_epochs)
    trainer.add_callback(eval_callback)
    if data_args.save_each_epochs:
        save_callback = CustomSaveCallback(save_each=data_args.save_each_epochs)
        trainer.add_callback(save_callback)

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        if not data_args.dont_save_in_the_end:
            trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        for k in eval_dataset_tokenized:
            metrics = {'EM {k}': eval_callback.em_score[k],
                       'F1 {k}': eval_callback.f1_score[k],
            }
            trainer.log_metrics(f"eval_{k}", metrics)
            trainer.save_metrics(f"eval_{k}", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

