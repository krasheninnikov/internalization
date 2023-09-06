# Out-of-context Meta-Learning in Large Language Models
This repo contains the code accompanying the paper Out-of-context Meta-Learning in Large Language Models. The codebase is build around Hugging Face Transformers' Trainer and contains implementations of multiple experiments discussed in the paper.

[![Tests](https://github.com/krasheninnikov/internalization/actions/workflows/main.yml/badge.svg)](https://github.com/krasheninnikov/internalization/actions/workflows/main.yml)

# Quickstart
1) Clone repository and run `bash setup.sh`. This will install the dependencies and download datasets.
2) From project root: `export PYTHONPATH="$PWD:${PYTHONPATH}"`. Alternatively, consider adding `export PYTHONPATH=$PYTHONPATH:/home/path/to/the/project/root` to your `.bashrc` file.
3) Choose/modify/create a config in `configs` folder (parameters description can be found here).
4) `python src/run.py --cp <your-config-path>` (default is `configs/current_experiment.yaml`).
