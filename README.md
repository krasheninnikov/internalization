# Internalization
[![Tests](https://github.com/krasheninnikov/internalization/actions/workflows/main.yml/badge.svg)](https://github.com/krasheninnikov/internalization/actions/workflows/main.yml)


1) From project root: `export PYTHONPATH="$PWD:${PYTHONPATH}"`. Alternatively, consider adding `export PYTHONPATH=$PYTHONPATH:/home/path/to/the/project/root` to your `.bashrc` file.
2) Modify config or create your own in `configs`.
3) `python src/run.py --config_name <your-config-name>` (default is `current_experiment.yaml`).