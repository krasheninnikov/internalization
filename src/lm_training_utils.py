import string
from itertools import product
from typing import Optional

import torch
from tokenizers import Tokenizer, pre_tokenizers
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


def create_tokenizer(add_tokens_for_var_names=True, num_letters_per_var=3, max_x=99):
    vocab = ["[PAD]", "[UNK]", "=", "%"]
    vocab += ['true', 'false', 'define1', 'define2']
    vocab += [str(i) for i in range(max_x+1)]  # numbers 0 to max_x get their own tokens
    vocab += list(string.ascii_lowercase)
    
    # add tokens for all possible variable names of length num_letters_per_var
    if add_tokens_for_var_names:
        var_name_tuples = list(product(*[string.ascii_lowercase]*num_letters_per_var))
        var_name_strings = ["".join(var_name_tuples[i]) for i in range(len(var_name_tuples))]
        vocab.extend(var_name_strings)

    str_to_tokid = {s: i for i, s in enumerate(vocab)}
    tokenizer = Tokenizer(WordLevel(str_to_tokid, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return tokenizer
