import string
from itertools import product
from typing import Optional

import torch
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
from torch.utils.data import RandomSampler, SequentialSampler
from transformers import Trainer
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm



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
    vocab += ['true', 'false', 'define1', 'define2', 'define3']
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


def linear_probe(hugginface_model, eval_dataset_d1, eval_dataset_d2):
    # prepare data, get representations from model
    train_reps = []
    train_labels = []
    for i in range(len(eval_dataset_d1)):
        inputs = eval_dataset_d1[i]
        with torch.no_grad():
            outputs = hugginface_model(**inputs)
            train_reps.append(outputs.last_hidden_state[:,-1,:].detach().numpy())
            train_labels.append(1)
            
    for i in range(len(eval_dataset_d2)):
        inputs = eval_dataset_d2[i]
        with torch.no_grad():
            outputs = hugginface_model(**inputs)
            train_reps.append(outputs.last_hidden_state[:,-1,:].detach().numpy())
            train_labels.append(0)
            
    train_reps = np.concatenate(train_reps, axis=0)
    train_labels = np.array(train_labels)
    
    # split data into train and test
    #X_train, X_test, y_train, y_test = train_test_split(train_reps, train_labels, test_size=0.2, random_state=42)
    
    # fit logistic regression model from statsmodels
    logit_model = sm.Logit(train_labels, sm.add_constant(train_reps))
    result = logit_model.fit()
    print(result.summary())
