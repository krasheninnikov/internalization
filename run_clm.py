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
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import pandas as pd
import datasets
import torch
from datasets import load_dataset
import numpy as np
import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    default_data_collator,
    set_seed,
    pipeline
)
from transformers.integrations import TensorBoardCallback
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from main import get_raw_datasets
from data_utils_define_experiment import get_questions_dataset, mixed_reliable_and_unreliable_data
from config import TAG
from metrics import compute_em_list, compute_f1_list
from trainer_no_shuffle_sampling import TrainerDeterministicSampler
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    paired_paragraphs: Optional[bool] = field(
        default=False, metadata={"help": "Whether the SQUAD paragraphs should be single paragraphs or concatenated "
                                         "pairs of paragraphs."}
    )
    define_experiment: Optional[bool] = field(
        default=False, metadata={"help": "Whether we perform the Define experiment. "
                                 "If False, paragraphs-as-insights experiment is performed."}
    )
    no_relevant_insights: Optional[bool] = field(
        default=False, metadata={"help": "The Define experiment where in the train set insights don't correspond to any questions"}
    )
    deterministic_sampler: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a deterministic sampler for training."}
    )
    append_insights_to_qs: Optional[bool] = field(
        default=False, metadata={"help": "Whether insights should be appended to questions or be separate datapoints."}
    )
    dataset: Optional[str] = field(
        default='squad', metadata={"help": "The name of the dataset to use (squad, archival, synth)."}
    )
    mix_reliable_unreliable_data: Optional[bool] = field(
        default=False, metadata={"help": "See mix_reliable_unreliable_data in data_utils_define_experiment.py"}
    )
    train_subset: Optional[str] = field(
        default='full', metadata={"help": "Param for the define experiment. One of (full, insights_ri, all_but_insights_ri)"}
    )
    synth_num_each_gender: Optional[int] = field(
        default=2000,
        metadata={"help": ("1/2 of the number of datapoints to generate; should be up to 60000 (so 120k total named entities);"
                           " can make much more with modifications but would need to make genders unbalanced")},
    )

    # Default run_clm args below
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        pass


class EvaluationCallback(TensorBoardCallback):
    def __init__(self, eval_dataset_raw, tb_writer=None):
        super(EvaluationCallback, self).__init__(tb_writer)
        self.eval_dataset_raw = eval_dataset_raw
        
    def on_epoch_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self.tb_writer is None:
            self._init_summary_writer(args)
        
        model.eval()
        tokenizer.padding_side = 'left'
        pipe = pipeline(task='text-generation', model=model, device=0, tokenizer=tokenizer)
        for k in self.eval_dataset_raw:
            logger.info(f'*** Evaluating on {k} ***')
            eval_dataset_k = self.eval_dataset_raw[k]
            original_answers = eval_dataset_k['answer']
            qa_prompts = eval_dataset_k['question']

            predicted_answers = pipe(qa_prompts,
                                    max_new_tokens=20,
                                    pad_token_id=tokenizer.pad_token_id,
                                    batch_size=args.per_device_eval_batch_size,
                                    clean_up_tokenization_spaces=True,
                                    return_full_text=False)
            
            predicted_answers = [x[0]['generated_text'].strip() for x in predicted_answers]
            em_score = compute_em_list(predicted_answers, original_answers)
            f1_score = compute_f1_list(predicted_answers, original_answers)

            self.tb_writer.add_scalar(f"eval/{k}_EM", em_score, state.global_step)
            self.tb_writer.add_scalar(f"eval/{k}_F1", f1_score, state.global_step)

        #results_df = pd.DataFrame(results, columns=['EM', 'F1'], index=inds)
        # print(results_df)
        


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.save_total_limit = 2
    # TODO figure out if line below is needed
    training_args.save_strategy = "no"
    training_args.load_best_model_at_end = True

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
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

    # experiment with replacing named entities with random strings
    print(f'Using dataset: {data_args.dataset}')
    if data_args.define_experiment:
        if data_args.mix_reliable_unreliable_data:
            raw_datasets = mixed_reliable_and_unreliable_data(seed=training_args.seed, 
                                                              train_subset=data_args.train_subset,
                                                              synth_num_each_gender=data_args.synth_num_each_gender,)
        
        elif data_args.no_relevant_insights:
            raw_datasets = get_questions_dataset(seed=training_args.seed,
                                                 frac_n_qri=0.0,
                                                 frac_n_qr=0.4,
                                                 frac_n_ri=0.25,
                                                 frac_n_r=0.1,
                                                 frac_n_q=0.25,
                                                 append_insights_to_qs=data_args.append_insights_to_qs,
                                                 dataset=data_args.dataset,
                                                 train_subset=data_args.train_subset,
                                                 synth_num_each_gender=data_args.synth_num_each_gender,)
        else:
            raw_datasets = get_questions_dataset(seed=training_args.seed,
                                                 append_insights_to_qs=data_args.append_insights_to_qs,
                                                 dataset=data_args.dataset,
                                                 train_subset=data_args.train_subset,
                                                 synth_num_each_gender=data_args.synth_num_each_gender,)
    # experiment with paragraphs and questions about them
    else:
        raw_datasets = get_raw_datasets(seed=training_args.seed, concat_pairs=data_args.paired_paragraphs)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
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

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # Added by Dima because GPT2 tokenizer doesn't have a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # tokenizer.add_special_tokens({'additional_special_tokens': [TAG]})
    # model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    # if training_args.do_train:
    #     column_names = raw_datasets["train"].column_names
    # else:
    #     column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        tokens = tokenizer(examples[text_column_name], padding='max_length', max_length=data_args.block_size, truncation=True)
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # TODO there must be a faster way to do this. This fn replaces the one above (group_texts)
    # def group_texts_alt(examples):
    #     examples["labels"] = examples["input_ids"].copy()
    #     return examples

    # with training_args.main_process_first(desc="grouping texts together"):
    #     lm_datasets = tokenized_datasets.map(
    #         group_texts_alt,
    #         batched=True,
    #         num_proc=data_args.preprocessing_num_workers,
    #         load_from_cache_file=not data_args.overwrite_cache,
    #         desc=f"Creating labels",
    #     )
    lm_datasets = tokenized_datasets

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    
    # evaluate on all datasets except the training set
    eval_dataset_keys = [k for k in lm_datasets]
    eval_dataset_keys.remove('train')
    if training_args.do_eval:
        eval_dataset = {key: lm_datasets[key] for key in eval_dataset_keys}
        eval_dataset_raw = {key: raw_datasets[key] for key in eval_dataset_keys}

        if data_args.max_eval_samples is not None:
            max_eval_samples = min(min([len(eval_dataset[k]) for k in eval_dataset]), data_args.max_eval_samples)
            # eval_dataset = eval_dataset.select(range(max_eval_samples))
            eval_dataset = {k: eval_dataset[k].select(range(max_eval_samples)) for k in eval_dataset}

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            # print(eval_preds[0])
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

        def generate_batch(examples):
            with torch.no_grad():
                input_ids = examples['input_ids'].to(training_args.device)
                attn_masks = examples['attention_mask'].to(training_args.device)
                outputs = model.generate(input_ids=input_ids,
                                         attention_mask=attn_masks,
                                         max_new_tokens=30, pad_token_id=tokenizer.pad_token_id)
            #decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return {'prediction': outputs}

    # Initialize our Trainer
    trainer_cls = TrainerDeterministicSampler if data_args.deterministic_sampler else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        # data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval else None,
    )
    trainer.pop_callback(TensorBoardCallback)
    eval_callback = EvaluationCallback(eval_dataset_raw)
    trainer.add_callback(eval_callback)

    # Training
    if training_args.do_train:
        training_args.evaluation_strategy = 'epoch'
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        model.to(training_args.device)
        model.eval()
        # for tokenizer to tokenize this column (contains only Q: <>\nA:)
        text_column_name = 'question'
        for k in eval_dataset_raw:
            eval_dataset_k = eval_dataset_raw[k]
            # with training_args.main_process_first(desc="dataset map tokenization"):
            #     eval_dataset_k = eval_dataset_k.map(
            #         tokenize_function,
            #         batched=True,
            #         num_proc=data_args.preprocessing_num_workers,
            #         remove_columns=column_names,
            #         load_from_cache_file=not data_args.overwrite_cache,
            #         desc="Running tokenizer on eval datasets",
            #     )
            #
            # predictions_k = eval_dataset_k.with_format('torch').map(
            #     generate_batch,
            #     batched=True,
            #     num_proc=data_args.preprocessing_num_workers,
            #     load_from_cache_file=not data_args.overwrite_cache,
            #     batch_size=100,
            #     remove_columns=['input_ids', 'attention_mask'],
            #     desc=f"Creating predictions for {k}",
            # )
            # decoded_outputs = tokenizer.batch_decode(predictions_k['prediction'], skip_special_tokens=True)
            # original_prompts = raw_datasets[k].select(range(0, 100))['text']
            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset_k['question'])
            max_eval_samples = min(max_eval_samples, len(eval_dataset_k['question']))
            original_answers = eval_dataset_k.select(range(max_eval_samples))['answer']
            qa_prompts = eval_dataset_k.select(range(max_eval_samples))['question']

            # predicted_answers = [x.replace(y, '') for x, y in zip(decoded_outputs, original_prompts)]
            # decoder-only models need left padding during generation
            tokenizer.padding_side = 'left'
            pipe = pipeline(task='text-generation', model=model, device=0, tokenizer=tokenizer)
            predicted_answers = pipe(qa_prompts,
                                     max_new_tokens=20,
                                     pad_token_id=tokenizer.pad_token_id,
                                     batch_size=training_args.per_device_eval_batch_size,
                                     num_workers=data_args.preprocessing_num_workers,
                                     clean_up_tokenization_spaces=True,
                                     return_full_text=False)
            predicted_answers = [x[0]['generated_text'].strip() for x in predicted_answers]

            # print example predictions and corresponding correct answers
            for i in range(3):
                print(f'Prompt: {qa_prompts[i]}')
                print(f'Correct & predicted answers: {original_answers[i], predicted_answers[i]}')
                print()

            # predicted_answers = [x[:x.find('\n')] for x in predicted_answers]
            # print(predicted_answers[:10])
            metrics = {'EM {k}': compute_em_list(predicted_answers, original_answers),
                       'F1 {k}': compute_f1_list(predicted_answers, original_answers),
                       "num_eval_samples {k}": max_eval_samples}

            # metrics = trainer.evaluate(eval_dataset[k])
            # try:
            #     perplexity = math.exp(metrics["eval_loss"])
            # except OverflowError:
            #     perplexity = float("inf")
            # metrics["perplexity"] = perplexity
            #
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


if __name__ == "__main__":
    main()
