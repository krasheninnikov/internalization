from datetime import datetime
import random
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.interpolate import interp1d

import torch as th
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader

import pytorch_lightning as pl
from fbm import fbm  # for generating fractional brownian motion data

from data_generation.data_utils import split_list_into_subsets


class Datapoint:
    def __init__(self, x, y, is_circle, is_triangle, is_square, cluster_center_idx, d_pos_embed=1):
        self.x = x
        self.y = y
        self.is_circle = is_circle
        self.is_triangle = is_triangle
        self.is_square = is_square
        self.cluster_center_idx = cluster_center_idx
        self.d_pos_embed = d_pos_embed
        
        self.min_x = 0
        self.max_x = 100000
        
    def get_features(self):
        def positional_embedding(pos, d):
            positions = np.array(pos).reshape(-1, 1)
            dimensions = np.arange(d).reshape(1, -1)
            div_term = 1 / np.power(100000, dimensions // 2 * 2 / d)
            embeddings = positions * div_term

            embeddings[:, ::2] = np.sin(embeddings[:, ::2])  # Apply sine to even dimensions
            embeddings[:, 1::2] = np.cos(embeddings[:, 1::2])  # Apply cosine to odd dimensions

            return embeddings

        one_hot_shape = np.array([self.is_circle, self.is_triangle, self.is_square], dtype=np.float32)
        
        # [PosEnc(x), 1, 0, 0] for circles, [PosEnc(x), 0, 1, 0] for triangles, [PosEnc(x), 0, 0, 1] for squares
        return np.concatenate([self.normalize_x(self.x, self.min_x, self.max_x) * one_hot_shape, 
                               positional_embedding(self.x, d=self.d_pos_embed).reshape(-1)])
        
        # Essentially [PosEnc(x), 0, 0] for circles, [0, PosEnc(x), 0] for triangles, [0, 0 PosEnc(x)] for squares
        return np.concatenate([self.normalize_x(self.x, self.min_x, self.max_x) * one_hot_shape, 
                               positional_embedding(self.x * one_hot_shape, d=self.d_pos_embed).reshape(-1)])
        # Just [x, 0, 0] for circles, [0, x, 0] for triangles, [0, 0, x] for squares
        # return self.normalize_x(self.x, self.min_x, self.max_x) * one_hot_shape

    
    def get_label(self):
        return self.y
    
    def __hash__(self) -> int:
        return hash((self.x, self.y, self.is_circle, self.is_triangle, self.is_square))
    
    def __repr__(self):
        return f'({self.x}, {self.y}, {self.is_circle}, {self.is_triangle}, {self.is_square})'

    @staticmethod
    def normalize_x(x, min_x, max_x):
        return (x - min_x) / (max_x - min_x)

    @staticmethod
    def unnormalize_x(x, min_x, max_x):
        return x * (max_x - min_x) + min_x


def uniform_interpolated_data(seed=0, n_anchors=20, n_interpolated_points=100000):
    np.random.seed(seed)
    x = np.arange(n_anchors)
    y = np.random.uniform(0, 1, n_anchors)
    x_new = np.linspace(x.min(), x.max(), n_interpolated_points)
    f = interp1d(x, y, kind='cubic')
    y_interp=f(x_new)
    return y_interp


def get_fractional_brownian_motion_data(hurst=.6, seed=0, n_points=100000):
    # TODO use seed
    # Generate a fBm realization
    return fbm(n=n_points, hurst=hurst, length=1, method='daviesharte')


def generate_data(hurst=.6, n_clusters = 400, cluster_spread = 200, n_datapoints_per_cluster = 50, seed=0, d_pos_embed=61):
    # data1 = get_fractional_brownian_motion_data(hurst=hurst, seed=seed)
    # data2 = get_fractional_brownian_motion_data(hurst=hurst, seed=seed*100)
    data1 = uniform_interpolated_data(seed=seed)
    data2 = uniform_interpolated_data(seed=(seed+1)*100)

    # normalize to [-1,1]
    data1 = (data1 - np.min(data1)) / (np.max(data1) - np.min(data1)) * 2 - 1
    data2 = (data2 - np.min(data2)) / (np.max(data2) - np.min(data2)) * 2 - 1
    # data1, data2 = data1*100, data2*100

    def sample_datapoint(cluster_center_index, cluster_spread, 
                        circle_noise_std=0, triangle_noise_std=0, square_noise_std=0): # noise stds are not used for now
        datapoint_idx = cluster_center_index + np.random.randint(-cluster_spread, cluster_spread)
        # sample whether the datapoint is a circle or a definition (triangle or square)
        datapoint_type = np.random.choice(['circle', 'definition'], p=[.75, .25])
        
        if datapoint_type == 'circle':
            x, y = datapoint_idx, data1[datapoint_idx]
            return Datapoint(x, np.random.normal(y, circle_noise_std), 1, 0, 0, cluster_center_index, d_pos_embed=d_pos_embed)
            
        elif datapoint_type == 'definition':
            # y vals for inconsistent definitions are sampled from data2, otherwise from data1
            if cluster_center_index in cluster_subsets['qd2incons']:
                x, y = datapoint_idx, data2[datapoint_idx]
            else:
                x, y = datapoint_idx, data1[datapoint_idx]
            
            # sample whether the definition is a triangle or a square (define1/define2)
            if cluster_center_index in cluster_subsets['qd1consis'].union(cluster_subsets['d1consis']):
                return Datapoint(x, np.random.normal(y, triangle_noise_std), 0, 1, 0, cluster_center_index, d_pos_embed=d_pos_embed)
            else:
                return Datapoint(x, np.random.normal(y, square_noise_std), 0, 0, 1, cluster_center_index, d_pos_embed=d_pos_embed)

    # randomly select indices for where the "clusters" would be
    cluster_center_indices = np.random.choice(np.arange(cluster_spread, len(data1)-cluster_spread), n_clusters, replace=False)
    # alternative: deterministic indices for where the "clusters" would be
    cluster_center_indices = np.arange(cluster_spread, len(data1)-cluster_spread, int(len(data1)/(n_clusters))).tolist()

    ###### split clusters into qd1consis, qd2incons, d1consis, d2consis ######
    
    # random.shuffle(cluster_center_indices_mid)
    # fracs_dict = {'qd1consis': .4, 'qd2incons': .4, 'd1consis': .1, 'd2consis': .1}

    print(f'total number of clusters: {len(cluster_center_indices)}')
    
    # Separate the middle 30% of the clusters (by x) from the rest: the middle 30% of the clusters (by x) should not have circles/qa pairs. 
    # Otherwise the circles can be inferred from their neighbors
    cluster_center_indices_mid = cluster_center_indices[int(len(cluster_center_indices)*.35):int(len(cluster_center_indices)*.65)]
    cluster_center_indices_excl_mid = [c for c in cluster_center_indices if c not in cluster_center_indices_mid]
    
    random.shuffle(cluster_center_indices_excl_mid)
    cluster_subsets_with_defs = split_list_into_subsets({'qd1consis': .5, 'qd2incons': .5,}, cluster_center_indices_excl_mid)
    
    # Randomly reverse the order of the middle 30% of the clusters (by x). 
    # This way we switch the x-wise order of triangle and square definitions -- sometimes no-QA triangles come before squares, sometimes after.
    if np.random.rand() > .5:
        cluster_center_indices_mid = cluster_center_indices_mid[::-1]
    cluster_subsets_wo_defs = split_list_into_subsets({'d1consis': .5, 'd2consis': .5,}, cluster_center_indices_mid)
    cluster_subsets = cluster_subsets_with_defs | cluster_subsets_wo_defs

    print(f"{[(k, len(cluster_subsets[k])) for k in cluster_subsets]}" )

    # sample datapoints from the clusters
    datapoints = [sample_datapoint(cluster_center_idx, cluster_spread) for cluster_center_idx in cluster_center_indices 
                  for _ in range(n_datapoints_per_cluster)]

    # take circles in d1consis and d2consis as test data and remove them from the datapoints list (that will become train data)
    test_sets = {'d1consis': [d for d in datapoints if d.is_circle and d.cluster_center_idx in cluster_subsets['d1consis']],
                 'd2consis': [d for d in datapoints if d.is_circle and d.cluster_center_idx in cluster_subsets['d2consis']]}
    # remove test data from the datapoints list
    datapoints = [d for d in datapoints if not (d.is_circle and d.cluster_center_idx in cluster_subsets['d1consis'].union(cluster_subsets['d2consis']))]

    # generate new qd1consis and qd2incons test data
    test_sets['qd1consis'] = [sample_datapoint(cluster_center_idx, cluster_spread) for _ in range(n_datapoints_per_cluster) for cluster_center_idx in cluster_subsets['qd1consis']]
    test_sets['qd2incons'] = [sample_datapoint(cluster_center_idx, cluster_spread) for _ in range(n_datapoints_per_cluster) for cluster_center_idx in cluster_subsets['qd2incons']]
    # leave only circles in qd1consis and qd2incons test data
    test_sets['qd1consis'] = [d for d in test_sets['qd1consis'] if d.is_circle]
    test_sets['qd2incons'] = [d for d in test_sets['qd2incons'] if d.is_circle]
    
    return datapoints, test_sets, data1, data2


class MLP(pl.LightningModule):
    def __init__(self, n_input_features=24, hidden_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input_features, hidden_size), nn.ReLU(), #nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), #nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), #nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), #nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 1)
        )
        self.l2 = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = self.l2(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return th.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-5)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.l2(y_hat, y)
        self.log(f"val_loss {dataloader_idx}", loss)


def get_tensor_dataset(data_list):
    x = th.Tensor(np.array([d.get_features() for d in data_list]))
    y = th.Tensor(np.array([d.get_label() for d in data_list])).unsqueeze(1)
    return TensorDataset(x,y)


if __name__ == '__main__':
    th.set_float32_matmul_precision('high')
    n_seeds = 200
    batch_size = 128
    epochs = 100
    
    run_name_suffix = 'sharedPosEnc_'
    run_name = f'toy_exp_{run_name_suffix}{datetime.now().strftime("%Y%m%d-%H%M%S")}_nseeds{n_seeds}_bs{batch_size}_epochs{epochs}'
    exp_folder = f'./toy_experiments/{run_name}'
    pathlib.Path(exp_folder).mkdir(parents=True, exist_ok=True)
    
    test_losses = {}
    for seed in range(n_seeds):
        datapoints, test_sets, data1, data2 = generate_data(seed=seed+400)
        print(f'total train datapoints: {len(datapoints)}')
        
        ####### plot the test/train datapoints and save to file #######
        # plot the train data
        plt.figure(figsize=(15, 5))
        plt.scatter([datapoint.x for datapoint in datapoints], [datapoint.y for datapoint in datapoints], 
                    c=['b' if datapoint.is_circle else 'g' if datapoint.is_triangle else 'r' for datapoint in datapoints])
        plt.plot(np.arange(len(data1)), data1, c = 'k')
        plt.plot(np.arange(len(data1)), data2, c = 'brown')
        plt.savefig(f'{exp_folder}/train_data_s{seed}.png')
        plt.clf()
        # plot the test data
        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(len(data2))/100000, data1, c = 'k')
        plt.plot(np.arange(len(data2))/100000, data2, c = 'brown')
        for k,v in test_sets.items():
            plt.scatter(np.sum(np.array([d.get_features() for d in v])[:, 0:3], 1), np.array([d.get_label() for d in v]), label=k)
        plt.legend()
        plt.savefig(f'{exp_folder}/test_data_s{seed}.png')

        ####### train the model #######        
        pl.seed_everything(seed)
        mlp = MLP(n_input_features=len(datapoints[0].get_features()))
        trainer = pl.Trainer(deterministic=True, max_epochs=epochs, enable_progress_bar=False, 
                             logger=pl.loggers.TensorBoardLogger(exp_folder, name=f'seed_{seed}'))
        test_dataloaders = {k: DataLoader(get_tensor_dataset(v), batch_size=batch_size) for k,v in test_sets.items()}

        trainer.fit(mlp, DataLoader(get_tensor_dataset(datapoints), batch_size=batch_size), val_dataloaders=test_dataloaders)
        
        # plot the model predictions as well as the underlying data
        plt.figure(figsize=(15, 5))
        plt.plot(np.arange(len(data2))/100000, data1, c = 'k')
        plt.plot(np.arange(len(data2))/100000, data2, c = 'brown')

        mlp.eval()
        with th.no_grad():
            test_losses[seed] = {}
            for k,v in test_sets.items():
                x = th.Tensor(np.array([d.get_features() for d in v]))
                y = th.Tensor(np.array([d.get_label() for d in v])).unsqueeze(1)
                out = mlp(x)
                loss = mlp.l2(out, y)
                print(f'{k} loss: {loss}')
                test_losses[seed][k] = loss.detach().numpy()
                # plot predictions
                plt.scatter(np.sum(x.detach().numpy()[:, 0:3], 1), out.detach().numpy(), label=k)
        plt.legend()
        plt.savefig(f'{exp_folder}/model_predictions_s{seed}.png')
        plt.clf()
        
        # plot a summary of the val losses as a barplot; this would be updated/overwritten every seed
        losses = {k: [float(v[k]) for v in test_losses.values()] for k in test_sets.keys()}
        plt.clf()    # clear the plot
        plt.figure(figsize=(15, 5))
        sns.barplot(data=pd.DataFrame(losses))
        plt.ylabel('MSE')
        plt.savefig(f'{exp_folder}/results.png')
