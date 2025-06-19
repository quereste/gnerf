import math
import logging
import numpy as np
import faiss
import faiss.contrib.torch_utils

import time
import torch
import torch.nn as nn
import torch.autograd.profiler as profiler

from typing import Optional
from utils.general_utils import append_sys_path
from gnerf.lagrangian_hash.knn.knn_algorithms import BaseKNN
import open3d as o3d


append_sys_path()

log = logging.getLogger(__name__)

class SplashEncoding(nn.Module):
    def __init__(
        self,
        fixed_std: bool = False,
        decay_factor: int = 1,
        n_gausses: int = 10000,
        n_features_per_gauss: int = 3,
        knn_algorithm: Optional[BaseKNN] = None
    ):
        """
        """
        super().__init__()
        assert knn_algorithm is not None, "KNN algorithm must be provided"
        
        self.decay_factor = decay_factor
        self.n_features_per_gauss = n_features_per_gauss

        r = 0.125
        self.total_gaus = self.init_mean()
        self.feats = (torch.randn(self.total_gaus, self.n_features_per_gauss) * 1e-2).to(device='cuda')
        self.feats = nn.Parameter(self.feats)
        self.means = nn.Parameter(self.means, requires_grad=False)
        self.gaussian_constant = torch.sqrt(torch.tensor(2 * torch.pi, device='cuda'))
        D = self.means.shape[1]
        cov_init = torch.eye(D, device='cuda').unsqueeze(0).repeat(self.total_gaus, 1, 1) * 0.1
        self.covariances = nn.Parameter(cov_init)
        self.knn = knn_algorithm
    
    # def init_mean(self):
        # N = self.total_gaus
        # log.info(f'Total number of gauss: {self.total_gaus}')
        # pts = np.random.randn(N, 3)
        # r = np.sqrt(np.random.rand(N, 1))
        # pts = pts / np.linalg.norm(pts, axis=1)[:, None] * r
        # pts = pts * 0.5 + 0.5 # [0.25 ... 0.75]
        
        # self.means = torch.tensor(pts, dtype=torch.float32, device='cuda')

    def init_mean(self):
        # Load .ply file and initialize means
        ply_path1 = "/workspace/gnerf/means_lod14@20000.ply"
        ply_path2 = "/workspace/gnerf/means_lod15@20000.ply"
        pcd1 = o3d.io.read_point_cloud(ply_path1)
        pcd2 = o3d.io.read_point_cloud(ply_path2)
        pts1 = np.asarray(pcd1.points)
        pts2 = np.asarray(pcd2.points)
        pts = np.concatenate([pts1, pts2], axis=0)

        # if pts.shape[0] > 40000:
        #     idx = np.random.choice(pts.shape[0], 40000, replace=False)
        #     pts = pts[idx]

        self.means = torch.tensor(pts, dtype=torch.float32, device='cuda')

        print(f"Loaded pointclud with {self.means.shape} points.")

        return self.means.shape[0]


    def update_factor(self):
        self.covariances = self.covariances * self.decay_factor


    def get_means(self):
        return self.means
    

    def set_means(self, means):
        means = means
        self.means = nn.Parameter(self.means)


    def get_covariances(self):
        return self.covariances
    

    def _calculate(self, coords, nearest_gausses_indicies, sq_dists, batch_size=100000):
        num_coords = coords.shape[0]
        feature_dim = self.feats.shape[1]
        D = coords.shape[1]
        eps = 1e-5

        feature_vector = torch.zeros((num_coords, feature_dim), device=coords.device)

        for i in range(0, num_coords, batch_size):
            batch_indices = nearest_gausses_indicies[i : i + batch_size]  # [B, K]
            batch_coords = coords[i : i + batch_size]  # [B, D]
            nearest_features = self.feats[batch_indices]  # [B, K, F]
            batch_factors = self.covariances[batch_indices]  # [B, K, D, D]
            batch_means = self.means[batch_indices]       # [B, K, D]

            eye = torch.eye(D, device=coords.device).expand(batch_factors.shape)
            batch_covs = torch.matmul(batch_factors, batch_factors.transpose(-2, -1)) + eps * eye  # [B, K, D, D]

            # Cholesky decomposition
            L = torch.linalg.cholesky(batch_covs)  # [B, K, D, D]
            diff = batch_coords[:, None, :] - batch_means  # [B, K, D]
            diff_unsq = diff.unsqueeze(-1)  # [B, K, D, 1]
            sol = torch.cholesky_solve(diff_unsq, L)  # [B, K, D, 1]
            mdist = torch.matmul(diff_unsq.transpose(-2, -1), sol).squeeze(-1).squeeze(-1)  # [B, K]

            # Determinant from Cholesky
            det = L.diagonal(dim1=-2, dim2=-1).prod(-1) ** 2  # [B, K]
            norm_const = torch.sqrt((2 * torch.pi) ** D * det + eps)

            gau_weights = torch.exp(-0.5 * mdist) / (norm_const + eps)  # [B, K]
            weighted_features = nearest_features * gau_weights.unsqueeze(-1)  # [B, K, F]
            batch_feature_vector = torch.sum(weighted_features, dim=1)  # [B, F]

            feature_vector[i : i + batch_size] = batch_feature_vector

        return feature_vector
        

    def forward(self, coords, lod_idx=None):
        batch_size = 5000000

        nearest_gausses_indicies = self.knn.get_nearest_neighbours(coords, self.means)

        # Calculate squared distance between each coord and its nearest mean
        nearest_means = self.means[nearest_gausses_indicies]
        squared_gausses_distance = torch.sum((coords[:, None, :] - nearest_means) ** 2, dim=-1)

        # start_time = time.time()
        feats = self._calculate(coords, nearest_gausses_indicies, squared_gausses_distance, batch_size=batch_size)
        # print(f"Features: {time.time() - start_time:.4f} seconds")

        return feats, squared_gausses_distance[:, 0]
    