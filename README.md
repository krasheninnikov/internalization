# Out-of-context Meta-Learning in Large Language Models

This repository contains the source code corresponding to the paper Out-of-context Meta-Learning in Large Language Models. The codebase is constructed around the Hugging Face Transformers' Trainer and includes implementations of various experiments described in the paper.

[![Tests](https://github.com/krasheninnikov/internalization/actions/workflows/main.yml/badge.svg)](https://github.com/krasheninnikov/internalization/actions/workflows/main.yml)

## Quickstart 

Get started with the codebase by following the steps below:

#### 1. Configure Python Environment
- **Step 1**: Create a new Conda environment. Replace "internalization" with the name you prefer for your environment:
  
   ```bash
   conda create --name internalization python=3.10
   ``` 
   Replace '3.10' with your desired version number.
  
- **Step 2**: Activate your Conda environment:
  
   ```bash
   conda activate internalization
   ```
  
- **Step 3**: You are now within your Conda environment where you can configure the PYTHONPATH specific to the project. Append the project root to PYTHONPATH in your activated Conda environment:
  
   ```bash
   export PYTHONPATH=/path/to/the/project/root:$PYTHONPATH
   ```
   
   Or from the project root:

   ```bash
   export PYTHONPATH="$PWD:${PYTHONPATH}"
   ```

#### 2. Clone Repository:

Start by cloning the repository using the following command in your terminal:
```bash
git clone https://github.com/krasheninnikov/internalization.git
```
Next, move into the newly cloned directory:
```bash
cd internalization
```
Install the necessary dependencies and download the datasets with the command:

```bash
bash setup.sh
```

#### 3. Choose/modify/create a Config:

Browse to the **configs** directory to select an existing configuration, modify as per your requirements, or create a new one. Further information related to parameter descriptions can be found in the [configs directory](./configs).

#### 4. Run the Experiment:

To run the experiment, use the following command: 

```python
python src/run.py --cp <your-config-path>
```
Please note that the default configuration is `configs/current_experiment.yaml`.