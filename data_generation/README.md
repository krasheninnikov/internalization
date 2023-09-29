# Understanding the data

See `demo.ipynb` for a quick overview of how to load our three text datasets (CVDB, T-REx, and set inclusion).


The following image should help relate our code to data subsets from the paper.

![Image](code-notation.png?raw=true "Code notation")


# `data_generation` Directory Overview

This `data_generation` directory contains a variety of essential scripts and modules crucial to the data generation and processing operations:

- `cvdb_data.py`: This script is responsible for processing the **cvdb** dataset.

- `trex_data.py`: This file handles data processing for the **trex** dataset.

- `data_objects.py`: This module contains principal objects such as Definitions, Questions, and QAPairs. These are used structurally in the generation of datasets for both 'define' and 'numeric' experiments.

- `define_experiment.py`: This script houses the primary function named `get_questions_dataset`. Its primary role is generating train and test sets for the 'define' experiment.

- `define_strings.py`: This file is a repository of templates used in experiments which involve natural language prompts and in-context definitions.

- `numerical_experiment.py`: This script is designed for generating datasets related to numerical experiments, which includes experiments involving modular divisions and number choice.

- `experiment.py`: This file executes the logic for loading datasets for specific experiments.

- `data_utils.py`: This file includes various helper functions that support different data-related tasks across the scripts in this directory.