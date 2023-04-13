import json
import os
import string
from itertools import product
from typing import Optional

import torch
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.implementations.base_tokenizer import BaseTokenizer
from tokenizers.models import WordLevel
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import Trainer


class TrainerDeterministicSampler(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # or not has_length(self.train_dataset):
        if self.train_dataset is None:
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        if self.args.world_size <= 1:
            # return RandomSampler(self.train_dataset, generator=generator)
            return SequentialSampler(self.train_dataset)  # Changed from above
        else:
            raise NotImplementedError(
                "Distributed training is not supported yet.")


class CharTokenizer(BaseTokenizer):
    def __init__(self, context_len, add_tokens_for_var_names=True, num_letters_per_var=3):
        self.ctx_len = context_len
        self.vocab = "[PAD],[UNK],=,%".split(",")
        self.vocab.extend([str(i) for i in range(100)])
        self.vocab.extend(list(string.ascii_lowercase))
        self.vocab.extend(['true', 'false', 'reliable', 'unreliable'])
        if add_tokens_for_var_names:
            var_name_tuples = list(
                product(*[string.ascii_lowercase]*num_letters_per_var))
            var_name_strings = ["".join(var_name_tuples[i])
                                for i in range(len(var_name_tuples))]
            self.vocab.extend(var_name_strings)

        self.str_to_tokid = {s: i for i, s in enumerate(self.vocab)}
        self.tokid_to_str = {i: s for i, s in enumerate(self.vocab)}

        self.PAD_TOK_ID = self.str_to_tokid["[PAD]"]
        self.UNK_TOK_ID = self.str_to_tokid["[UNK]"]

        self.pad_token_id = self.PAD_TOK_ID
        self.unk_token_id = self.UNK_TOK_ID
        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"

        tokenizer = Tokenizer(WordLevel(self.str_to_tokid, unk_token='[UNK]'))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer.enable_truncation(max_length=self.ctx_len)
        tokenizer.enable_padding(
            pad_token="[PAD]", pad_id=self.PAD_TOK_ID, length=self.ctx_len, direction="right")
        parameters = {
            "model": "WordLevel",
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
        }

        super().__init__(tokenizer, parameters)

    @property
    def vocab_size(self):
        return len(self.vocab)


def create_tokenizer(add_tokens_for_var_names=True, num_letters_per_var=3):
    vocab = "[PAD],[UNK],=,%".split(",")
    vocab += [str(i) for i in range(100)]
    vocab += list(string.ascii_lowercase)
    vocab += ['true', 'false', 'reliable', 'unreliable']
    if add_tokens_for_var_names:
        var_name_tuples = list(
            product(*[string.ascii_lowercase]*num_letters_per_var))
        var_name_strings = ["".join(var_name_tuples[i])
                            for i in range(len(var_name_tuples))]
        vocab.extend(var_name_strings)

    str_to_tokid = {s: i for i, s in enumerate(vocab)}
    tokenizer = Tokenizer(WordLevel(str_to_tokid, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tokenizer