# Internalization
[![Tests](https://github.com/krasheninnikov/internalization/actions/workflows/main.yml/badge.svg)](https://github.com/krasheninnikov/internalization/actions/workflows/main.yml)

0) Clone repository and run `bash setup.sh`. This will install the dependencies and download datasets.
1) From project root: `export PYTHONPATH="$PWD:${PYTHONPATH}"`. Alternatively, consider adding `export PYTHONPATH=$PYTHONPATH:/home/path/to/the/project/root` to your `.bashrc` file.
2) Modify config or create your own in `configs`.
3) `python src/run.py --config_path <your-config-path>` (default is `configs/current_experiment.yaml`).