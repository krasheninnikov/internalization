# `src` Directory Overview

- `finetuning.py`: contains the code used to construct the experiment pipelines of different types including 1-stage, 2-stage, and 3-stage, for both natural language and set inclusion experiments.

- `callbacks.py`: contais the callbacks for custom evaluation and gradient alignment experiments.

- `metrics.py`: code for computing Exact Match (EM), F1 Score, etc.

- `train_lm.py`: main script for training/finetuning a language model.

- `lm_training_utils`: contains a custom tokenizer for the set inclusion experiment, and a sequential data sampler.

- `run.py`: entry point for running multiple seeds and launching SLURM jobs.