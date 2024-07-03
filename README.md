# Implicit meta-learning may lead language models to trust more reliable sources

[![Tests](https://github.com/krasheninnikov/internalization/actions/workflows/main.yml/badge.svg)](https://github.com/krasheninnikov/internalization/actions/workflows/main.yml)

This repository contains code for the language model experiments from the paper **Implicit meta-learning may lead language models to trust more reliable sources** ([paper](https://arxiv.org/abs/2310.15047), [old poster](https://drive.google.com/file/d/1aZMzo8Dzz20FIoxKhgsY62bjSp-LEuH9/view)). 

Steps to get started:


### 1. Clone the repository

```bash
git clone https://github.com/krasheninnikov/internalization.git
cd internalization
```


### 2. Configure your Python environment
- **Step 1.** Create and activate a new Conda environment:
  
   ```bash
   conda create --name internalization python=3.11
   conda activate internalization
   ``` 

- **Step 2.** Install the dependencies and download the datasets:

   ```bash
   pip install -r requirements.txt
   # download the datasets from Google Drive
   gdown --folder 'https://drive.google.com/drive/folders/1KQDClI3cbFzPhzfknF2xmtqE-aIW1EDf?usp=sharing'
   ```

- **Step 3 (Optional).**
   Configure `wandb`:
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
python -m src.run -cp <your-config-path>
```
