# Configuration Parameters Documentation
This README describes the configurable experiment parameters that are classified under different categories, and how to override these parameters at the various (finetuning) stages of the experiment.

## Argument Categories
The specification parameters are classified into five different categories. You can find more details in the respective dataclass inside [utils/arguments.py](../utils/arguments.py):
1. `training_arguments`: These arguments are related to `ModelTrainingArguments`.
2. `data_arguments`: These arguments are associated with `DataTrainingArguments`.
3. `model_arguments`: These arguments are related to `ModelArguments`.
4. `experiment_arguments`: These belong to `CommonExperimentArguments`.
5. `define_experiment_arguments`: These include arguments from `DefineExperimentDataArguments`.
6. `numeric_experiment_arguments`: These arguments are a part of `NumericExperimentDataArguments`.

## Argument Overrides 
For each stage of your experiment, you can override arguments using the following configurations:
1. `first_stage_arguments`: This is an overriding dictionary for stage one of model training/finetuning, accepting parameters from different argument groups.
2. `second_stage_arguments`: This overriding dictionary is meant for stage two of finetuning.
3. `third_stage_arguments`: This overriding dictionary is used for stage three.

The total number of stages can be set in the `experiment_arguments`. Parameters for each stage are overridden using the respective dictionary. 

In an experiment with only one stage, the `first_stage_arguments` override parameters from other argument groups. In a two-stage experiment, parameters for the first stage are overridden using `first_stage_arguments`, while `second_stage_arguments` are used for the second stage.