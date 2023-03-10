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
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import torch
import transformers
from transformers import (CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING,
                          AutoConfig, AutoModelForCausalLM,
                          AutoModelForSeq2SeqLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, HfArgumentParser,
                          MaxLengthCriteria, PreTrainedTokenizerFast,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          StoppingCriteriaList, Trainer, default_data_collator,
                          set_seed)
from transformers.integrations import TensorBoardCallback
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry

from callbacks import EvaluationCallbackGenerate, EvaluationCallbackPipeline, CustomSaveCallback
from data_scripts.numeric_experiment import *
from data_scripts.data_utils_define_experiment import get_questions_dataset
from data_scripts.squad_data import get_raw_datasets
from logger import setup_logger
from utils import CharTokenizer, TrainerDeterministicSampler

logger = setup_logger(__name__)


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
    max_new_tokens: int = field(
        default=20,
        metadata={
            "help": (
                "Max number of new tokens to generate. "
            )
        }
    )
    seq2seq: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether seq2seq model is going to be used or not. "
            )
        }
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
        default=True, metadata={"help": "Whether we perform the Define experiment. "
                                 "If False, paragraphs-as-defns experiment is performed."}
    )
    numeric_experiment: Optional[bool] = field(
        default=False, metadata={"help": "Whether we perform the toy numeric experiment. "}
    )
    modular_experiment: Optional[bool] = field(
        default=False, metadata={"help": "Whether we use baseline data for the Modular experiment. "}
    )
    modular_experiment_baseline: Optional[bool] = field(
        default=False, metadata={"help": "Whether we use baseline data for the Modular experiment. "}
    )
    num_choice_experiment: Optional[bool] = field(
        default=False, metadata={"help": "Num choice experiment. "}
    )
    no_relevant_defns: Optional[bool] = field(
        default=False, metadata={"help": "The Define experiment where in the train set defns don't correspond to any questions"}
    )
    deterministic_sampler: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use a deterministic sampler for training."}
    )
    dataset: Optional[str] = field(
        default='cvdb', metadata={"help": "The name of the dataset to use (cvdb, squad, archival)."}
    )
    mix_reliable_unreliable_data: Optional[bool] = field(
        default=True, metadata={"help": "See mix_reliable_unreliable_data in data_utils_define_experiment.py"}
    )
    train_subset: Optional[str] = field(
        default='full', metadata={"help": ("Param for the define experiment. "
                                           "One of (full, stage1, stage2, stage1_only_defns, stage1_only_qa)")}
    )
    num_ents: Optional[int] = field(
        default=4000,
        metadata={"help": ("number of ents used to generate the data to generate; should be up to 120k for cvdb;"
                           " can make much more with modifications but would need to make genders unbalanced")},
    )
    def_order: Optional[str] = field(
        default='tve', metadata={"help": "The order of Tag, Variable and Entity in definitions."}
    )
    seed_stage2: Optional[int] = field(
        default=0,
        metadata={"help": ("Seed for the data split of stage 2 (d1consis, d2consis, no_qd_baseline)")},
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
        metadata={"help": (
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
        default=1024, metadata={"help": ("Optional input sequence length after tokenization. "
                                         "The training dataset will be truncated in block of this size for training. "
                                         "Default to the model max input length for single sentence inputs (take into account special tokens).")},
    )
    label_block_size: Optional[int] = field(
        default=48, metadata={"help": ("Optional input sequence length after tokenization. "
                                       "The training dataset will be truncated in block of this size for training. "
                                       "Default to the model max input length for single sentence inputs (take into account special tokens).")},
    )
    save_each_epochs: Optional[int] = field(
        default=None, metadata={"help": ("Make a checkpoint each `save_each_epochs`")}
    )
    
    eval_each_epochs: Optional[int] = field(
        default=1, metadata={"help": "Perform evaluation every eval_each_epochs which calculates EM/F1"}
    )
    
    dont_save_in_the_end: Optional[bool] = field(
        default=False, metadata={"help": "Don't save the model in the end."}
    )
    
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5, metadata={"help": "The percentage of the train set used as validation set in case there's no validation split"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True, metadata={"help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."},
    )


@dataclass
class NumericExperimentDataArguments:
    """
    Arguments pertaining to the num_choice experiment.
    """
    max_x: Optional[int] = field(default=99, metadata={"help": ("")},)
    num_x: Optional[int] = field(default=500, metadata={"help": ("")},)
    n_nums_in_question: Optional[int] = field(default=4, metadata={"help": ("")},)
    n_intersecton: Optional[int] = field(default=2, metadata={"help": ("")},)
    n_qs_per_x: Optional[int] = field(default=2*12, metadata={"help": ("")},)
    p_label_flip: Optional[float] = field(default=0.0, metadata={"help": ("")},)
    

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, NumericExperimentDataArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, numeric_exp_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, numeric_exp_args, training_args = parser.parse_args_into_dataclasses()

    # training_args.save_total_limit = 2
    training_args.save_strategy = "no"
    training_args.load_best_model_at_end = False
    training_args.evaluation_strategy = 'epoch'


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
    logger.info(f'Using dataset: {data_args.dataset}')
    if data_args.define_experiment:
        if data_args.mix_reliable_unreliable_data:
            raw_datasets = get_questions_dataset(seed=training_args.seed,
                                                 seed_stage2=data_args.seed_stage2,
                                                 frac_n_qd1consis=0.25,
                                                 frac_n_qd2incons=0.25,
                                                 frac_n_q=0.1,
                                                 frac_n_d1consis=0.1,
                                                 frac_n_d2consis=0.1,
                                                 frac_n_no_qd_baseline=0.1,
                                                 frac_n_q_no_replacement_baseline=0.1,
                                                 dataset_name=data_args.dataset,
                                                 train_subset=data_args.train_subset,
                                                 num_ents=data_args.num_ents,
                                                 def_order=data_args.def_order,)
            
        elif data_args.no_relevant_defns:
            raw_datasets = get_questions_dataset(seed=training_args.seed,
                                                 seed_stage2=data_args.seed_stage2,
                                                 frac_n_qd1consis=0.0,
                                                 frac_n_qd2incons=0.0,
                                                 frac_n_q=0.4,
                                                 frac_n_d1consis=0.25,
                                                 frac_n_d2consis=0.0,
                                                 frac_n_no_qd_baseline=0.1,
                                                 frac_n_q_no_replacement_baseline=0.25,
                                                 dataset_name=data_args.dataset,
                                                 train_subset=data_args.train_subset,
                                                 num_ents=data_args.num_ents,
                                                 def_order=data_args.def_order,)
        else:
            raw_datasets = get_questions_dataset(seed=training_args.seed,
                                                 seed_stage2=data_args.seed_stage2,
                                                 dataset_name=data_args.dataset,
                                                 train_subset=data_args.train_subset,
                                                 num_ents=data_args.num_ents,
                                                 def_order=data_args.def_order,)
    elif data_args.numeric_experiment:
        if data_args.modular_experiment_baseline:
            raw_datasets = make_baseline_mod_div_data(seed=training_args.seed,
                                                      train_subset=data_args.train_subset,)

        elif data_args.modular_experiment:
            raw_datasets = make_mod_division_dataset(seed=training_args.seed,
                                                     train_subset=data_args.train_subset,)
            
        elif data_args.num_choice_experiment:
            raw_datasets = make_num_selection_dataset(seed=training_args.seed,
                                                      train_subset=data_args.train_subset,
                                                      max_x=numeric_exp_args.max_x,
                                                      num_x=numeric_exp_args.num_x,
                                                      n_nums_in_question=numeric_exp_args.n_nums_in_question,
                                                      n_intersecton=numeric_exp_args.n_intersecton,
                                                      n_qs_per_x=numeric_exp_args.n_qs_per_x,
                                                      p_label_flip=numeric_exp_args.p_label_flip,)
        else:
            raise ValueError('Must specify a numeric experiment type (num_choice_experiment, modular_experiment, or modular_experiment_baseline)')
    # experiment with paragraphs and questions about them
    else:
        raw_datasets = get_raw_datasets(seed=training_args.seed, concat_pairs=data_args.paired_paragraphs)

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
    if data_args.numeric_experiment:
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
    if data_args.numeric_experiment:
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
    eval_callback = EvaluationCallbackPipeline(eval_dataset_raw, numeric_experiment=data_args.numeric_experiment, eval_each=data_args.eval_each_epochs)
    #eval_callback = EvaluationCallback(eval_dataset_tokenized, generate_batch, postprocess_output_fn=postprocess_output_fn, numeric_experiment=data_args.numeric_experiment, eval_each=data_args.eval_each_epochs)
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


if __name__ == "__main__":
    main()
