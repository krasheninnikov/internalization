# Contents

The `data_generation` directory includes essential scripts and modules, integral to data generation and processing:

- `cvdb_data.py`: This file handles the data processing tasks for the **cvdb** dataset.

- `trex_data.py`: This script is devoted to the data processing tasks of the **trex** dataset.

- `data_objects.py`: This module encompasses the principal objects employed to logically structure the datasets generation process, including Definitions, Questions, and QAPairs for both 'define' and 'numeric' experiments.

- `define_experiment.py`: This file underscores the primary function `get_questions_dataset` - responsible for creating the train and test sets for the define experiment.

- `define_strings.py`: This script houses templates for experiments involving natural language prompts and in-context definitions.

- `numerical_experiment.py`: This file is utilized for conducting numerical experiments, encompassing modular division and number choice experiments.

- `experiment.py`: This script implements the logic of loading datasets for specific experiments.

- `data_utils.py`: 