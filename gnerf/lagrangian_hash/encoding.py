import numpy as np

import torch
import torch.nn as nn

from typing import Optional
from utils.general_utils import append_sys_path
from gnerf.lagrangian_hash.knn.knn_algorithms import BaseKNN

append_sys_path()


class SplashEncoding(nn.Module):
    def __init__(
        self,
        decay_factor: int = 1,
        n_gausses: int = 10000,
        n_features_per_gauss: int = 3,
        knn_algorithm: Optional[BaseKNN] = None,
        means: Optional[np.ndarray] = None,
    ):
        """
        """
        super().__init__()
        assert knn_algorithm is not None, "KNN algorithm must be provided"
        
        self.decay_factor = decay_factor
        self.n_features_per_gauss = n_features_per_gauss

        if means is not None:
            self.means = torch.tensor(means, dtype=torch.float32, device='cuda')
        else:
            self.means = self.init_mean(n_gausses)

        # if self.means.shape[0] > 40000:
        #     idx = np.random.choice(self.means.shape[0], 40000, replace=False)
        #     self.means = self.means[idx]

        self.means = nn.Parameter(self.means, requires_grad=False)

        self.total_gaus = self.means.shape[0]
        self.feats = (torch.randn(self.total_gaus, self.n_features_per_gauss) * 1e-2).to(device='cuda')
        self.feats = nn.Parameter(self.feats)
        self.gaussian_constant = torch.sqrt(torch.tensor(2 * torch.pi, device='cuda'))
        self.log_diag_cov_2d = nn.Parameter(torch.log(torch.ones(self.total_gaus, 2, device='cuda') * 0.01))
        self.knn = knn_algorithm

    
    def init_mean(self, N):
        print(f'Total number of gauss: {N}')
        pts = np.random.randn(N, 3)
        r = np.sqrt(np.random.rand(N, 1))
        pts = pts / np.linalg.norm(pts, axis=1)[:, None] * r
        pts = pts * 0.5 + 0.5 # [0.25 ... 0.75]
        
        return torch.tensor(pts, dtype=torch.float32, device='cuda')


    def get_means(self):
        return self.means
    

    def set_means(self, means):
        means = means
        self.means = nn.Parameter(self.means)


    def get_covariances(self):
        return self.covariances
    

    def densify(self, gradients, step, max_steps):
        """
        Add new Gaussians based on gradient direction, following the original
        Gaussian Splatting paper (Kerbl et al., 2023).

        Args:
            gradients (Tensor): [N, 3], gradients w.r.t. position (e.g. features or means).
            step (int): Current training step.
            max_steps (int): Max number of training steps.
        """
        device = self.means.device
        means = self.means.detach()
        new_means = []

        # Scheduled threshold (exponential interpolation)
        grad_thresh_min = 1e-4
        grad_thresh_max = 4e-4
        alpha = step / float(max_steps)
        grad_thresh = grad_thresh_min * (grad_thresh_max / grad_thresh_min) ** alpha

        # Jitter config (used instead of step-based offset)
        jitter_scale = 0.005  # or similar small value

        for i in range(means.shape[0]):
            grad = gradients[i]
            g_pos = means[i]

            grad_norm = torch.norm(grad)
            if grad_norm > grad_thresh:
                jitter = torch.randn(2, 3, device=device) * jitter_scale
                new_pos = g_pos[None, :] + jitter  # [2, 3]
                new_means.append(new_pos)

        if new_means:
            new_means = torch.cat(new_means, dim=0)  # [M, 3]
            self.append_means(new_means)


    def append_means(self, new_means):
        new_feats = torch.randn(new_means.shape[0], self.n_features_per_gauss, device=new_means.device) * 1e-2
        new_covs = torch.log(torch.ones(new_means.shape[0], 2, device=new_means.device) * 0.01)

        was_trainable = self.means.requires_grad
        self.means = nn.Parameter(torch.cat([self.means.detach(), new_means], dim=0), requires_grad=was_trainable)
        self.feats = nn.Parameter(torch.cat([self.feats.detach(), new_feats], dim=0))
        self.log_diag_cov_2d = nn.Parameter(torch.cat([self.log_diag_cov_2d.detach(), new_covs], dim=0))


    def unfreeze_means(self):
        self.means.requires_grad_(True)


    def freeze_means(self):
        self.means.requires_grad_(False)
        

    def _calculate(self, coords, nearest_gausses_indicies, sq_dists, batch_size=100000):
        num_coords = coords.shape[0]
        feature_dim = self.feats.shape[1]
        D = coords.shape[1]  # Should be 3
        eps = 1e-5

        feature_vector = torch.zeros((num_coords, feature_dim), device=coords.device)

        for i in range(0, num_coords, batch_size):
            batch_indices = nearest_gausses_indicies[i : i + batch_size]  # [B, K]
            batch_coords = coords[i : i + batch_size]  # [B, 3]
            nearest_features = self.feats[batch_indices]  # [B, K, F]

            # Get the first two diagonal elements (trainable)
            batch_log_diag_2d = self.log_diag_cov_2d[batch_indices]  # [B, K, 2]
            batch_diag_2d = torch.exp(batch_log_diag_2d) + eps
            batch_diag = torch.cat([batch_diag_2d, torch.full_like(batch_diag_2d[..., :1], 1e-6)], dim=-1)
            
            batch_means = self.means[batch_indices]  # [B, K, 3]

            diff = batch_coords[:, None, :] - batch_means      # [B, K, 3]
            mdist = (diff ** 2 / batch_diag).sum(-1)           # [B, K]

            # Normalization constant for diagonal Gaussian
            norm_const = torch.sqrt((2 * torch.pi) ** D * batch_diag.prod(-1) + eps)  # [B, K]

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
    