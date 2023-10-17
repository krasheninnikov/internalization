# `src` Directory Overview

**Three main scripts:**
- `run.py`: entry point for running multiple seeds and launching SLURM jobs

- `experiment_pipeline.py`: code for constructing experiment pipelines for 1/2/3-stage finetuning, for both natural language and set inclusion experiments

- `train_lm.py`: main script for training/finetuning a language model

**Others:**
- `callbacks.py`: callbacks for custom evaluation and gradient alignment experiments

- `metrics.py`: code for computing Exact Match (EM), F1 Score, etc

- `lm_training_utils`: contains a custom tokenizer for the set inclusion experiment, and a sequential data sampler
