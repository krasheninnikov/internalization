# Directory Overview (`data_generation`)

This directory contains data generation and processing code. See [`demo.ipynb`](demo.ipynb) for a quick overview of how to load our three text datasets (CVDB, T-REx, and set inclusion).

**Three main scripts:**
- `load_data_from_config.py`: Logic for loading data for a specified experiment configuration.

- `define_experiment.py`: Generating train and test sets for the main natural language experiment.

- `numerical_experiment.py`: Generating the data for the set inclusion experiment.

**Others:**
- `data_objects.py`: Contains objects such as Definitions, Questions, and QAPairs. These are used when generating data for both 'define' and 'set inclusion / numeric' experiments.

- `cvdb_data.py`: Data processing for the **cvdb** dataset.

- `trex_data.py`: Data processing for the **trex** dataset.

- `define_strings.py`: Templates for experiments with natural language prompts (e.g. "According to Wikipedia, xyz means Cleopatra") and in-context definitions.

- `data_utils.py`


This image should help relate our code to data subsets from the paper:

![Image](code-notation.png?raw=true "Code notation")
