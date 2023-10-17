# Configuration Parameters Documentation
This README describes parameter categories and how to override these parameters at the various (finetuning) stages of the experiment.

## Argument Categories
Parameters in each configuration `yaml` file are divided into six categories / groups. You can find more details in the respective dataclasses inside [utils/arguments.py](../utils/arguments.py):
1. `training_arguments`: see `ModelTrainingArguments` in `utils/arguments.py`.
2. `data_arguments`: see `DataTrainingArguments`.
3. `model_arguments`: see `ModelArguments`.
4. `experiment_arguments`: see `CommonExperimentArguments`.
5. `define_experiment_arguments`: see `DefineExperimentDataArguments`.
6. `numeric_experiment_arguments`: see `NumericExperimentDataArguments`.

## Argument Overrides 
For each stage of finetuning, you can override the parameters above using the following parameter groups:
1. `first_stage_arguments`: This is an overriding dictionary for stage one of model training/finetuning, accepting parameters from different argument groups.
2. `second_stage_arguments`: This overriding dictionary is meant for stage two of finetuning.
3. `third_stage_arguments`: This overriding dictionary is used for stage three.

For example,  you can set the parameter `num_train_epochs` (normally part of `training_arguments`) to 20 for the first stage and 10 for the second stage (see how this is done in [configs/current_experiment.yaml](../configs/current_experiment.yaml)).

The number of finetuning stages (1, 2, or 3) can be set using the `n_stages` parameter in the `experiment_arguments` parameter group.