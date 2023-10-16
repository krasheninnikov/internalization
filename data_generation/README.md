# Directory Overview (`data_generation`)

This directory contains code crucial to the data generation and processing operations. See [`demo.ipynb`](demo.ipynb) for a quick overview of how to load our three text datasets (CVDB, T-REx, and set inclusion).

**Three main scripts:**
- `experiment.py`: Logic for loading data for a specified experiment configuration.

- `define_experiment.py`: Generating train and test sets for the main natural language experiment.

- `numerical_experiment.py`: Generating the data for the set inclusion experiment.

**Other scripts:**
- `data_objects.py`: Contains objects such as Definitions, Questions, and QAPairs. These are used in the generation of datasets for both 'define' and 'set inclusion / numeric' experiments.

- `cvdb_data.py`: Handles data processing for the **cvdb** dataset.

- `trex_data.py`: Handles data processing for the **trex** dataset.

- `define_strings.py`: Includes templates for experiments which involve natural language prompts and in-context definitions.

- `data_utils.py`


This image should help relate our code to data subsets from the paper:

![Image](code-notation.png?raw=true "Code notation")
