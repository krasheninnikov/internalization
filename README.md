# Implicit meta-learning may lead language models to trust more reliable sources


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

- **Step 2.** Install the dependencies and download the data:

   ```bash
   pip install -r requirements.txt
   mkdir -p datasets/cvdb  # make a folder for the dataset
   ```
   Download the CVDB dataset from https://data.sciencespo.fr/dataset.xhtml?persistentId=doi:10.21410/7E4/RDAG3O# and unzip `cross-verified-database.csv` into the folder above.

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
