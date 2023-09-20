# `src` Directory Overview
The `src` directory contains essential scripts and modules that are integral to the functioning and execution of this project. Here is a breakdown of what each file does:

- `finetuning.py`: This file contains the code required to construct the experiment pipelines of different types including 1-stage, 2-stage, and 3-stage.

- `callbacks.py`: This script has the callbacks necessary for custom evaluation and gradient alignment experiments.

- `metrics.py`: Contains the Evaluation Metrics modules - Exact Match (EM), F1 Score, etc.

- `train_lm.py`: This file is a script for the training model.

- `lm_training_utils`: It features custom tokenizer and sequential sampler.

- `run.py`: This script serves as the entry point for running multiple seeds and launching SLURM jobs.