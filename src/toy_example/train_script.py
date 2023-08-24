from datetime import datetime
from src.toy_example.toy_data_generation import generate_data, get_tensor_dataset, MLP
import pathlib
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import ttest_ind

import torch as th
from torch.utils.data import DataLoader

import pytorch_lightning as pl
import argparse
import wandb
import os
from utils.logger import setup_logger

logger = setup_logger(__name__)


wandb_config = {'project': 'internalization',
                'entity': 'assistance-llms', 
                'notes': os.environ.get('SLURM_JOB_ID', 'local')}



def train(config=None):
    run = wandb.init(config=config, **wandb_config)
    args = run.config
        
    n_anchors = args.n_clusters#args.n_anchors
    batch_size = args.batch_size
    epochs = args.epochs
    hidden_size = args.hidden_size
    n_seeds = args.n_seeds
    d_y = args.d_y
    max_x = args.max_x
    n_clusters = args.n_clusters
    cluster_spread = args.cluster_spread
    d_pos_enc = args.d_pos_enc
    n_datapoints_per_cluster = args.n_datapoints_per_cluster
    p_definition = args.p_definition
    
    logger.info(args)

    featurization = 'separateQaDefChannels' # one of ["singleChannel", "separateQaDefChannels", "3separateChannels"]
    
    run_name_suffix = ''
    run_name = (f'toy_exp_{run_name_suffix}{datetime.now().strftime("%Y%m%d-%H%M%S")}'
                f'_{featurization}_dy{d_y}_nAnchors{n_anchors}_bs{batch_size}_epochs{epochs}_nnWidth{hidden_size}')
    exp_folder = f'./toy_experiments/{run_name}'
    pathlib.Path(exp_folder).mkdir(parents=True, exist_ok=True)

    config_dict = {'n_seeds': n_seeds, 'batch_size': batch_size, 'epochs': epochs, 'd_y': d_y, 'max_x': max_x, 'n_anchors': n_anchors, 
                   'featurization': featurization, 'n_clusters': n_clusters, 'cluster_spread': cluster_spread,
                   'n_datapoints_per_cluster': n_datapoints_per_cluster, 'p_definition': p_definition, 'd_pos_enc': d_pos_enc,}
    json.dump(config_dict, open(f'{exp_folder}/config.json', 'w'))

    test_losses = {}
    for seed in range(n_seeds):
        train_datapoints, test_sets, data1, data2 = generate_data(seed=seed+400, n_anchors=n_anchors, n_datapoints=max_x, d_y=d_y, featurization=featurization,
                                                                  n_clusters=n_clusters, cluster_spread=cluster_spread, n_datapoints_per_cluster=n_datapoints_per_cluster,
                                                                  p_definition=p_definition, d_pos_enc=d_pos_enc)
        
        print(f'total train datapoints: {len(train_datapoints)}')
        
        ####### plot the test/train datapoints and save to file #######
        # plot the train data
        # TODO use different markers for circles/triangles/squares instead of colors
        plt.figure(figsize=(15, 5))
        plt.scatter([d.x_normalized for d in train_datapoints], [d.get_label()[0] for d in train_datapoints], 
                    c=['gray' if d.is_circle else 'green' if d.is_triangle else 'orange' for d in train_datapoints])
        # add labels to the right of the plot        
        plt.text(1.03, 0.9, 'circles', color='gray', transform=plt.gca().transAxes)
        plt.text(1.03, 0.85, 'triangles', color='green', transform=plt.gca().transAxes)
        plt.text(1.03, 0.8, 'squares', color='orange', transform=plt.gca().transAxes)
        plt.title(f'train data, seed {seed}')
        plt.plot(np.arange(len(data1))/max_x, data1[:, 0], c = 'k')
        plt.plot(np.arange(len(data2))/max_x, data2[:, 0], c = 'brown')
        plt.savefig(f'{exp_folder}/train_data_s{seed}.png')
        plt.clf()
        # plot the test data with the same color palette as in QA experiments
        color2order = {'blue': 0, 'orange': 1, 'green': 2, 'red': 3, 'purple': 4, 'brown': 5, 'pink': 6, 'gray': 7, 'olive': 8, 'cyan': 9}  
        name2color = {'d1consis': 'blue', 'q': 'brown',  'qd2incons': 'pink',  'd2consis': 'red', 'qd1consis': 'purple',
                  'no_qd_baseline': 'orange', 'q_no_replacement_baseline': 'green', 'qd1incons': 'cyan', 'qd2consis': 'olive', 'd3consis': 'gray'}
        palette = sns.color_palette()  # default palette, muted version of tab10
        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(len(data1))/max_x, data1[:, 0], c = 'k')
        plt.plot(np.arange(len(data2))/max_x, data2[:, 0], c = 'brown')
        for subset_name, data in test_sets.items():
            plt.scatter(np.array([d.x_normalized for d in data]), np.array([d.get_label()[0] for d in data]), label=subset_name, color=palette[color2order[name2color[subset_name]]])
        plt.legend()
        plt.title(f'test data, seed {seed}')
        plt.savefig(f'{exp_folder}/test_data_s{seed}.png')
        
        wandb.log({'plot_train_data': [wandb.Image(f'{exp_folder}/train_data_s{seed}.png')],
                   'plot_test_data': [wandb.Image(f'{exp_folder}/test_data_s{seed}.png')]})
        

        ####### train the model #######    
        th.set_float32_matmul_precision('high')
        pl.seed_everything(seed)
        mlp = MLP(n_in=len(train_datapoints[0].get_features()), n_out=len(train_datapoints[0].get_label()), hidden_size=hidden_size)
        trainer = pl.Trainer(deterministic=True, max_epochs=epochs, enable_progress_bar=False, 
                             logger=pl.loggers.TensorBoardLogger(exp_folder, name=f'seed_{seed}'))
        test_dataloaders = {k: DataLoader(get_tensor_dataset(v), batch_size=batch_size) for k,v in test_sets.items()}

        trainer.fit(mlp, DataLoader(get_tensor_dataset(train_datapoints), batch_size=batch_size), val_dataloaders=test_dataloaders)     
        
        # plot the model predictions as well as the underlying data     
        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(len(data2))/max_x, data1[:, 0], c = 'k')
        plt.plot(np.arange(len(data2))/max_x, data2[:, 0], c = 'brown')

        mlp.eval()
        with th.no_grad():
            test_losses[seed] = {}
            for subset_name, data in test_sets.items():
                x = th.Tensor(np.array([d.get_features() for d in data]))
                y = th.Tensor(np.array([d.get_label() for d in data])) #.unsqueeze(1)                
                y_hat = mlp(x)

                dim_to_keep_matrix = th.Tensor(np.array([d.one_hot_dim_to_keep for d in data]))
                loss = mlp.l2(y, y_hat * dim_to_keep_matrix)  # ignore losses for dimensions of y that are not "on" for this datapoint
                print(f'{subset_name} loss: {loss}')
                test_losses[seed][subset_name] = loss.detach().numpy()
                # plot predictions; NOTE that we don't plot those where d.dim_to_keep != 0
                dim_to_keep_is_0_idx = [i for i, d in enumerate(data) if d.dim_to_keep==0]
                plt.scatter(np.array([d.x_normalized for d in data])[dim_to_keep_is_0_idx], y_hat.detach().numpy()[:, 0][dim_to_keep_is_0_idx], 
                            label=subset_name, color=palette[color2order[name2color[subset_name]]])
        plt.legend()
        plt.savefig(f'{exp_folder}/model_predictions_s{seed}.png')
        plt.clf()
        
        wandb.log({'plot_model_predictions': [wandb.Image(f'{exp_folder}/model_predictions_s{seed}.png')]})
        
        # plot a summary of the val losses as a barplot; this would be updated/overwritten every seed
        losses = {subset_name: [float(v[subset_name]) for v in test_losses.values()] for subset_name in test_sets.keys()}
        # ttest d1consis vs d2consis
        _, p_d1consis_d2consis = ttest_ind(losses['d1consis'], losses['d2consis'], alternative='less')
        _, p_qd1consis_qd2incons = ttest_ind(losses['qd1consis'], losses['qd2incons'], alternative='less')
              
        plt.clf()    # clear the plot
        plt.figure(figsize=(15, 5))
        sns.barplot(data=pd.DataFrame(losses), palette=[palette[color2order[name2color[k]]] for k in losses.keys()])
        plt.title(f'p(qd1consis < qd2incons) = {p_qd1consis_qd2incons:.4f}, p(d1consis < d2consis) = {p_d1consis_d2consis:.4f}, n_seeds = {len(losses["d1consis"])}')
        plt.ylabel('MSE')
        plt.savefig(f'{exp_folder}/results.png')
        
        # save means, stds, n_seeds, p-values, etc in a results.json file
        result_dict = {'n_seeds': len(losses['d1consis']),
                'd1consis': {'mean': np.mean(losses['d1consis']), 'std': np.std(losses['d1consis'])},
                'd2consis': {'mean': np.mean(losses['d2consis']), 'std': np.std(losses['d2consis'])},
                'qd1consis': {'mean': np.mean(losses['qd1consis']), 'std': np.std(losses['qd1consis'])},
                'qd2incons': {'mean': np.mean(losses['qd2incons']), 'std': np.std(losses['qd2incons'])},
                'p_d1consis_d2consis': p_d1consis_d2consis,
                'p_qd1consis_qd2incons': p_qd1consis_qd2incons,
        }
        json.dump(result_dict, open(f'{exp_folder}/results.json', 'w'))
        
    metric = -np.mean(losses['qd1consis']) + np.mean(losses['qd2incons']) - np.mean(losses['d1consis']) + np.mean(losses['d2consis'])  # maximize this
    wandb.log({'metric': metric, 'd1consis': np.mean(losses['d1consis']), 'd2consis': np.mean(losses['d2consis']), 'qd1consis': np.mean(losses['qd1consis']),
                   'qd2incons': np.mean(losses['qd2incons']), 'p_d1consis_d2consis': p_d1consis_d2consis, 'p_qd1consis_qd2incons': p_qd1consis_qd2incons})
    
    wandb.log(
        {'plot_MSE': [wandb.Image(f'{exp_folder}/results.png')]}
        )
    run.finish()
    return metric
    

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--sweep_id", type=str, help="Sweep ID for wandb", required=True)
    # args = parser.parse_args()

    # sweep_id = args.sweep_id
    # wandb.agent(sweep_id, function=train, entity=wandb_config['entity'], project=wandb_config['project'])
    train()