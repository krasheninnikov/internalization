# Meta- (out-of-context) Learning in Neural Networks 

[![Tests](https://github.com/krasheninnikov/internalization/actions/workflows/main.yml/badge.svg)](https://github.com/krasheninnikov/internalization/actions/workflows/main.yml)

This repository contains the source code for the paper *Meta- (out-of-context) Learning in Neural Networks*. The codebase implements language model experiments described in the paper, and relies heavily on the HuggingFace Transformers library.

Follow these steps to get started:


### 1. Clone the repository

In your terminal, enter:
```bash
git clone https://github.com/krasheninnikov/internalization.git
cd internalization
```


### 2. Configure your Python environment
- **Step 1**: Create a new Conda environment. Replace "internalization" with the name you prefer for your environment, and "3.11" with the desired Python version:
  
   ```bash
   conda create --name internalization python=3.11
   ``` 
  
- **Step 2**: Activate your Conda environment:
  
   ```bash
   conda activate internalization
   ```

- **Step 3**: Install the necessary dependencies and download the datasets with the command:

   ```bash
   bash setup.sh
   ```

   Configure `wandb` (optional):
   ```bash
   wandb login
   wandb init --entity=your-entity --project=your-project
   ```
  

### 3. Run the experiment

To run the experiment with the default configuration ([`configs/current_experiment.yaml`](./configs/current_experiment.yaml)), use the following command: 

```python
python -m src.run
```

**Choosing/modifying/creating an experiment configuration.** Go to the [**configs**](./configs) directory to select an existing configuration or create a new one. Some parameter descriptions can be found in the [configs readme](./configs/README.md). 

Once the configuration is ready, run the experiment with the following command:
```python
python -m src.run --cp <your-config-path>
```