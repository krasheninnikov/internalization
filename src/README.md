This folder contains:

* `finetuning.py` -- the code used to create the experiment pipelines of different types (1-3 stage).
* `callbacks.py` -- callbacks for custom evaluation and gradient alignment experiments.
* `metrics.py` -- metrics used for evaluation (EM/F1).
* `train_lm.py` -- the training script.
* `lm_training_utils` -- custom tokenizer and sequential sampler.
* `run.py` -- the entrypoint for running multiple seeds and launching SLURM jobs.
