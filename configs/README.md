# Configuration Parameters Documentation
In this documentation, you will learn about the configurable parameters that are classified under different categories, and how to override these parameters at various stages of your experiment.

## Argument Categories
The specification parameters are classified into five different categories. You can find more details in the respective dataclass inside [arguments.py](../utils/arguments.py):
1. **training_arguments**: These arguments are related to `ModelTrainingArguments`.
2. **data_arguments**: These arguments are associated with `DataTrainingArguments`.
3. **model_arguments**: These arguments are related to `ModelArguments`.
4. **experiment_arguments**: These belong to `CommonExperimentArguments`.
5. **define_experiment_arguments**: These include arguments from `DefineExperimentDataArguments`.
6. **numeric_experiment_arguments**: These arguments are a part of `NumericExperimentDataArguments`.

## Argument Overrides 
For each stage of your experiment, you can override arguments using the following configurations:
1. **first_stage_arguments**: This is an overriding dictionary for stage one, accepting parameters from different argument groups.
2. **second_stage_arguments**: This overriding dictionary is meant for stage two.
3. **third_stage_arguments**: This overriding dictionary is used for stage three.

The total number of stages can be set in the `experiment_arguments`. Parameters for each stage are overridden using the respective dictionary. 

In an experiment with only one stage, the **first_stage_arguments** override parameters from other argument groups. In a two-stage experiment, parameters for the first stage are overridden using **first_stage_arguments**, while **second_stage_arguments** is used for the second stage.