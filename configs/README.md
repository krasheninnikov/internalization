# Configuration Parameters Documentation
## Classification of Argument Groups
This guide offers an understanding of specification parameters that are separated into five distinct categories:
* `training_arguments`: ModelTrainingArguments
* `data_arguments`: DataTrainingArguments
* `experiment_arguments`: CommonExperimentArguments
* `define_experiment_arguments`: DefineExperimentDataArguments
* `numeric_experiment_arguments`: NumericExperimentDataArguments

Note: Detailed explanations for each argument category, as well as the parameters they consist of, are available in their respective dataclass.

## Argument Overrides
You can also override arguments for each stage of your experiment, which is done via the following:
* `first_stage_arguments`: The overriding dictionary for stage one, allows inclusion of parameters from diverse argument groups.
* `second_stage_arguments`: The overriding dictionary for stage two.
* `third_stage_arguments`: The overriding dictionary for stage three.

The number of stages is defined within `experiment_arguments`. Each stage will utilise a separate overwriting dictionary. For example, if there is only one stage, then parameter values from `first_stage_arguments` are used to override parameter values from the argument groups. 

In a two-stage experiment, parameters for the first stage use `first_stage_arguments` as its dictionary for override, while `second_stage_arguments` dictionary is used for overriding parameters in the second stage.