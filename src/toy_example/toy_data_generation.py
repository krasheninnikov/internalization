import random
from typing import Dict, List, Set, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch as th
from scipy.interpolate import interp1d
from torch import nn
from torch.utils.data import TensorDataset

from data_generation.data_utils import split_list_into_subsets


class Datapoint:
    def __init__(self, x, y, is_circle, is_triangle, is_square, cluster_center_idx, d_pos_enc=1, featurization="singleChannel"):
        self.x = x
        self.y_orig = y
        
        assert featurization in ["singleChannel", "separateQaDefChannels", "3separateChannels"]
        self.featurization = featurization
        
        self.is_circle = is_circle
        self.is_triangle = is_triangle
        self.is_square = is_square
        self.one_hot_shape = np.array([self.is_circle, self.is_triangle, self.is_square], dtype=np.float32)
        
        self.cluster_center_idx = cluster_center_idx
        self.d_pos_enc = d_pos_enc
        
        self.min_x = 0
        self.max_x = 100000
        self.x_normalized = self.normalize_x(self.x, self.min_x, self.max_x)
        
        self.y=y
        self.dim_to_keep = 0
        self.one_hot_dim_to_keep = np.ones((1,))
        if len(self.y)>1:
            self.one_hot_dim_to_keep = np.ones(len(self.y))
            
            # randomy set all but one dimension of y to -10          
            if self.is_circle:
                self.dim_to_keep = np.random.randint(0, len(self.y))
                self.one_hot_dim_to_keep = np.zeros(len(self.y))
                self.one_hot_dim_to_keep[self.dim_to_keep] = 1                
                self.y = self.y * self.one_hot_dim_to_keep  # set all but dim_to_keep index of y to 0


    def get_features(self):
        def positional_encoding(pos: Union[int, float, np.ndarray], d: int) -> np.ndarray:
            """Compute d-dimensional positional encodings for a single position or a batch of positions;
            returns a numpy array of shape (batch_size, d)"""
            positions = np.array(pos).reshape(-1, 1)
            dimensions = np.arange(d).reshape(1, -1)
            div_term = 1 / np.power(100000, dimensions // 2 * 2 / d)
            embeddings = positions * div_term

            embeddings[:, ::2] = np.sin(embeddings[:, ::2])  # Apply sine to even dimensions
            embeddings[:, 1::2] = np.cos(embeddings[:, 1::2])  # Apply cosine to odd dimensions

            return embeddings
        
        # [PosEnc(x), 1, 0, 0] for circles, [PosEnc(x), 0, 1, 0] for triangles, [PosEnc(x), 0, 0, 1] for squares
        if self.featurization == 'singleChannel':
            return np.concatenate([self.one_hot_shape,
                                   self.one_hot_dim_to_keep,
                                   positional_encoding(self.x, self.d_pos_enc).reshape(-1)])  # d-dimensional vector
        
        # Essentially [PosEnc(x), 0, 0] for circles, [0, PosEnc(x), 0] for triangles, [0, 0 PosEnc(x)] for squares
        elif self.featurization == '3separateChannels':
        # This seems to work even with d_y=1???????
            return np.concatenate([self.one_hot_shape, 
                                   self.one_hot_dim_to_keep,
                                   positional_encoding(self.x * self.one_hot_shape, self.d_pos_enc).reshape(-1)]) # (3*d)-dimensional vector

        # PosEnc(x) is in the same channel for triangles and squares, but in a different channel for circles        
        elif self.featurization == 'separateQaDefChannels':
            return np.concatenate([self.one_hot_shape, 
                                   self.one_hot_dim_to_keep,
                                   positional_encoding(self.x * np.array([self.is_circle, self.is_triangle or self.is_square], dtype=np.float32), 
                                                       self.d_pos_enc).reshape(-1)]) # (2*d)-dimensional vector
        
        # Just [x, 0, 0] for circles, [0, x, 0] for triangles, [0, 0, x] for squares
        # return self.x_normalized * self.one_hot_shape
    
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


def uniform_interpolated_data(seed=0, n_anchors=20, n_interpolated_points=100000, d=1, normalize=True, interp_kind='zero') -> np.ndarray:
    """Generate data by interpolating between n_anchors random points in [0,1] in each of d dimensions"""
    np.random.seed(seed)
    y_per_dim = np.zeros((n_interpolated_points, d), dtype=np.float32)
    x = np.arange(n_anchors)
    # if interp_kind == 'zero':
    #     x = np.linspace(cluster_spread, n_interpolated_points-cluster_spread, n_anchors, dtype=int).tolist()
    x_interp = np.linspace(min(x), max(x), n_interpolated_points)
    for i in range(d):
        y = np.random.uniform(0, 1, n_anchors)
        f = interp1d(x, y, kind=interp_kind)
        y_interp = f(x_interp)
        if normalize:     # normalize to [-1,1]
            y_interp = (y_interp - y_interp.min()) / (y_interp.max() - y_interp.min()) * 2 - 1
        y_per_dim[:, i] = y_interp
    return y_per_dim


def get_fractional_brownian_motion_data(hurst=.6, seed=0, n_points=100000):
    # TODO use seed
    # Generate a fBm realization
    from fbm import fbm  # for generating fractional brownian motion data
    return fbm(n=n_points, hurst=hurst, length=1, method='daviesharte')


def select_cluster_centers(data_len, n_clusters=400, cluster_spread=200, seed=0) -> Dict[str, Set[int]]:
    """select indices for where the "clusters" would be"""
    #cluster_center_indices = np.random.choice(np.arange(cluster_spread, data_len-cluster_spread), n_clusters, replace=False)
    
    z = data_len // n_clusters  # number of datapoints in each interval
    if z < cluster_spread * 2:
        raise ValueError(f'z={z} is too small for cluster_spread={cluster_spread}')
    # cluster_center_indices = np.linspace(cluster_spread, data_len-cluster_spread, n_clusters, dtype=int).tolist()
    cluster_center_indices = np.linspace(z // 2, data_len-z//2, n_clusters - 1, dtype=int).tolist()
    # select cluster centers such that they are not too close to the edges
    
    print(f'Total number of clusters: {len(cluster_center_indices)}')

    ###### split clusters into qd1consis, qd2incons, d1consis, d2consis ######
    # random.shuffle(cluster_center_indices_mid)
    # fracs_dict = {'qd1consis': .4, 'qd2incons': .4, 'd1consis': .1, 'd2consis': .1}
    
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
    return cluster_subsets


def generate_data(n_datapoints=100000, n_clusters = 400, cluster_spread = 200, n_datapoints_per_cluster = 50, seed=0, 
                  d_pos_enc=61, hurst=.6, n_anchors=20, d_y=1, featurization='singleChannel', p_definition=.25):
    # data1 = get_fractional_brownian_motion_data(hurst=hurst, seed=seed)
    # data2 = get_fractional_brownian_motion_data(hurst=hurst, seed=seed*100)
    data1 = uniform_interpolated_data(seed=seed, n_interpolated_points=n_datapoints, d=d_y, n_anchors=n_anchors, cluster_spread=cluster_spread)
    data2 = uniform_interpolated_data(seed=(seed+1)*100, n_interpolated_points=n_datapoints, d=d_y, n_anchors=n_anchors, cluster_spread=cluster_spread)

    assert len(data1) == len(data2) == n_datapoints
    cluster_subsets = select_cluster_centers(data_len=len(data1), n_clusters=n_clusters, cluster_spread=cluster_spread, seed=seed)
    print(f"Cluster subset lengths: {[(k, len(cluster_subsets[k])) for k in cluster_subsets]}")

    ###### sample datapoints from the clusters ######
    def sample_datapoint(cluster_center_index, cluster_spread, 
                         circle_noise_std=0, triangle_noise_std=0, square_noise_std=0): # noise stds are not used for now
        datapoint_idx = cluster_center_index + np.random.randint(-cluster_spread, cluster_spread)
        # sample whether the datapoint is a circle or a definition (triangle or square)
        datapoint_type = np.random.choice(['circle', 'definition'], p=[1-p_definition, p_definition])
        
        x = datapoint_idx
        if datapoint_type == 'circle':
            y = data1[datapoint_idx]
            return Datapoint(x, np.random.normal(y, circle_noise_std), 1, 0, 0, cluster_center_index, d_pos_enc=d_pos_enc, featurization=featurization)
            
        elif datapoint_type == 'definition':
            # y vals for inconsistent definitions are sampled from data2, otherwise from data1
            y = data2[datapoint_idx] if cluster_center_index in cluster_subsets['qd2incons'] else data1[datapoint_idx]
            
            # sample whether the definition is a triangle or a square (define1/define2)
            if cluster_center_index in cluster_subsets['qd1consis'].union(cluster_subsets['d1consis']):
                return Datapoint(x, np.random.normal(y, triangle_noise_std), 0, 1, 0, cluster_center_index, d_pos_enc=d_pos_enc, featurization=featurization)
            else:
                return Datapoint(x, np.random.normal(y, square_noise_std), 0, 0, 1, cluster_center_index, d_pos_enc=d_pos_enc, featurization=featurization)

    cluster_center_indices_all = [c for c_list in cluster_subsets.values() for c in c_list]
    datapoints = [sample_datapoint(cluster_center_idx, cluster_spread) for cluster_center_idx in cluster_center_indices_all 
                  for _ in range(n_datapoints_per_cluster)]

    # take circles in d1consis and d2consis as test data and remove them from the datapoints list (that will become train data)
    test_sets = {'d1consis': [d for d in datapoints if d.is_circle and d.cluster_center_idx in cluster_subsets['d1consis']],
                 'd2consis': [d for d in datapoints if d.is_circle and d.cluster_center_idx in cluster_subsets['d2consis']]}
    # remove test data from the datapoints list
    datapoints = [d for d in datapoints if not (d.is_circle and d.cluster_center_idx in cluster_subsets['d1consis'].union(cluster_subsets['d2consis']))]

    # generate new qd1consis and qd2incons test data
    n_test_datapoints_per_cluster = n_datapoints_per_cluster * d_y  # TODO should we do this upsampling?
    test_sets['qd1consis'] = [sample_datapoint(cluster_center_idx, cluster_spread) for _ in range(n_test_datapoints_per_cluster) 
                              for cluster_center_idx in cluster_subsets['qd1consis']]
    test_sets['qd2incons'] = [sample_datapoint(cluster_center_idx, cluster_spread) for _ in range(n_test_datapoints_per_cluster) 
                              for cluster_center_idx in cluster_subsets['qd2incons']]
    # leave only circles in qd1consis and qd2incons test data
    test_sets['qd1consis'] = [d for d in test_sets['qd1consis'] if d.is_circle]
    test_sets['qd2incons'] = [d for d in test_sets['qd2incons'] if d.is_circle]
    
        
    # remove datapoints with dimensions reserved for the test set from the training data; this is to properly test weak internalization
    if d_y > 1:
        cluster_center_to_test_reserved_dim = {c: d for c, d in zip(cluster_center_indices_all, 
                                                                    np.random.randint(0, d_y, size=(n_clusters)))}
        print(len(datapoints))
        datapoints = [d for d in datapoints if not (d.is_circle and d.dim_to_keep == cluster_center_to_test_reserved_dim[d.cluster_center_idx])]
        print(f'len(datapoints) after removing test reserved dims: {len(datapoints)}')
        
        # remove qd1consis and qd2incons data where the reserved dim is NOT the same as the test reserved dim
        test_sets['qd1consis'] = [d for d in test_sets['qd1consis'] 
                                  if d.dim_to_keep == cluster_center_to_test_reserved_dim[d.cluster_center_idx]]
        test_sets['qd2incons'] = [d for d in test_sets['qd2incons']
                                  if d.dim_to_keep == cluster_center_to_test_reserved_dim[d.cluster_center_idx]]
        
    
    return datapoints, test_sets, data1, data2


class MLP(pl.LightningModule):
    def __init__(self, n_in=24, n_out=1, hidden_size=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_in, hidden_size), nn.ReLU(), #nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), #nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), #nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), #nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, n_out)
        )
        self.l2 = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.l2(self.forward(x), y)
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
    y = th.Tensor(np.array([d.get_label() for d in data_list])) #.unsqueeze(1)
    return TensorDataset(x,y)    
