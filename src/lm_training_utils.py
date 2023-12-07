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


def linear_probe(hugginface_model, eval_dataset_d1, eval_dataset_d2, save_path='logit_results.txt', device='cuda'):
    try:
        import statsmodels.api as sm
    except ImportError:
        print("Please install statsmodels to run this function.")
        return
    
    # select only dict with keys 'input_ids', 'attention_mask', 'labels' and transfer to device
    def generate_repr(batch):
        batch = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        with torch.no_grad():
            outputs = hugginface_model(**batch, output_hidden_states=True)
            return {'representation': outputs.hidden_states[-1][:, -1, :]}
        
    eval_dataset_d1 = eval_dataset_d1.remove_columns([col for col in eval_dataset_d1.column_names if col not in ['input_ids', 'attention_mask']])
    eval_dataset_d2 = eval_dataset_d2.remove_columns([col for col in eval_dataset_d1.column_names if col not in ['input_ids', 'attention_mask']])


    preds_d1 = eval_dataset_d1.with_format('torch', device=device).map(
                generate_repr,
                batched=True,
                remove_columns=['input_ids', 'attention_mask'],
                batch_size=32,
                desc=f"Creating representations for d1",
            )

    preds_d2 = eval_dataset_d2.with_format('torch', device=device).map(
                generate_repr,
                batched=True,
                remove_columns=['input_ids', 'attention_mask'],
                batch_size=32,
                desc=f"Creating representations for d2",
            )
    
    # Concatenate predictions
    reps = torch.cat((preds_d1['representation'], preds_d2['representation']), dim=0)

    # Create labels
    labels_d1 = torch.ones(preds_d1['representation'].shape[0])
    labels_d2 = torch.zeros(preds_d2['representation'].shape[0])
    labels = torch.cat((labels_d1, labels_d2), dim=0)

    # Convert to numpy arrays
    reps = reps.cpu().numpy()
    labels = labels.cpu().numpy()
    print(reps.shape, labels.shape)
  
    # fit logistic regression model from statsmodels
    reps = sm.add_constant(reps)

    # from sklearn.decomposition import PCA

    # # Define the PCA object
    # pca = PCA(n_components=2)  # You can change the number of components

    # # Fit and transform the data
    # reps_pca = pca.fit_transform(reps)

    # Now use the transformed data in your logistic regression model
    logit_model = sm.Logit(labels, reps)
    result = logit_model.fit()
    print(result.summary())

    # write results to file
    with open(save_path, "w") as f:
        f.write(result.summary().as_text())