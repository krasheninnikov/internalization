import torch
from torch.utils.data import DataLoader
from transformers import (TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2TokenizerFast)
from abc import ABCMeta, abstractmethod
import os
from typing import List, Dict, Iterable
import numpy as np
from config import *


class Model(metaclass=ABCMeta):
    """Basic model template"""
    def __init__(self, name=None):
        self.name = name

    @abstractmethod
    def fit(self, dataset_dict):
        """Fit the model"""
        return

    @abstractmethod
    def evaluate(self, validation_sets):
        """Evaluate the model"""
        return

    @abstractmethod
    def generate(self, prediction_dataset):
        """Predict labels"""
        return


class GPT2Model(Model):
    def __init__(self, name='gpt2'):
        super(GPT2Model, self).__init__(name)
        self.name = name
        self.estimator = AutoModelForCausalLM.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        #special_tokens_dict = {'additional_special_tokens': [TAG]}
        #num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)
        #self.estimator.resize_token_embeddings(len(self.tokenizer))
        # select device (gpu/cpu)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print('Mode: {}'.format(self.device.type.upper()))
        # send estimator to device
        self.estimator.to(self.device)

    def fit(self, dataset_dict):
        """
        Train the model on given data for NUM_EPOCHS epochs (see config.py).
        @param train_dataset: training dataset, contains input_ids and attention_mask.
        @param eval_dataset: evaluation dataset.
        """
        train_dataset = dataset_dict['train']
        train_dataset = train_dataset.map(lambda x: self.tokenizer(x['text'],
                                                                   padding='max_length',
                                                                   max_length=1024,
                                                                   return_tensors='pt'),
                                          batched=True, remove_columns=['question', 'answer', 'text'])
        train_dataset = train_dataset.add_column("label", train_dataset['input_ids'].copy())
        train_dataset = train_dataset.with_format(type="torch",
                                                  columns=["input_ids", "attention_mask", "label"],
                                                  device=self.device)

        # initialize training arguments
        self.training_args = TrainingArguments(output_dir=f'results/{self.name}_results',
                                               num_train_epochs=NUM_EPOCHS,
                                               per_device_train_batch_size=BATCH_SIZE,
                                               per_device_eval_batch_size=BATCH_SIZE,
                                               warmup_steps=WARMUP_STEPS,
                                               weight_decay=WEIGHT_DECAY,
                                               logging_steps=10,
                                               evaluation_strategy="epoch",
                                               save_strategy="no",
                                               gradient_checkpointing=False,
                                               gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                                               )
        # initialize trainer
        trainer = Trainer(model=self.estimator, args=self.training_args,
                          train_dataset=train_dataset, tokenizer=self.tokenizer)
        trainer.train()

    def generate(self, encodings):
        with torch.no_grad():
            generated_ids = self.estimator.generate(**encodings)

    def evaluate(self, validation_sets):
        pass