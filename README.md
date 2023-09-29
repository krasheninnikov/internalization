# Meta- (out-of-context) Learning in Neural Networks

This repository contains the source code for the paper *Meta- (out-of-context) Learning in Neural Networks*. The codebase is constructed around the Hugging Face Transformers' Trainer and includes implementations of the language model experiments described in the paper.

<!-- ## Quickstart  -->

Get started with the codebase by following the steps below:


### 1. Clone the repository (NOTE: THIS WON'T WORK WITH THE ANONYMOUS REPO):

Start by cloning the repository using the following command in your terminal:
```bash
git clone https://github.com/krasheninnikov/internalization.git
```
Next, move into the newly cloned directory:
```bash
cd internalization
```


### 2. Configure your Python Environment
- **Step 1**: Create a new Conda environment. Replace "internalization" with the name you prefer for your environment:
  
   ```bash
   conda create --name internalization python=3.10
   ``` 
   Replace '3.10' with your desired version number.
  
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
  
- **Step 4**: Append the project root to PYTHONPATH in your activated Conda environment (alternatively, just add the command below to your `~/.bashrc` file):
  
   ```bash
   export PYTHONPATH=/path/to/the/project/root:$PYTHONPATH
   ```
   
   Or from the project root:

   ```bash
   export PYTHONPATH="$PWD:${PYTHONPATH}"
   ```

### 3. Choose/modify/create an experiment configuration:

Browse to the **configs** directory to select an existing configuration, modify as per your requirements, or create a new one. Further information related to parameter descriptions can be found in the [configs directory](./configs).

### 4. Run the Experiment:

To run the experiment, use the following command: 

```python
python src/run.py --cp <your-config-path>
```
Please note that the default configuration is `configs/current_experiment.yaml`.