"""PaDiM (Patch Distribution Modeling) implementation for PCB Defect Anomaly Detection.

Reference:
    Defard et al., "PaDiM: a Patch Distribution Modeling Framework for
    Anomaly Detection and Localization", ICPR 2020.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class PaDiM(nn.Module):
    """Patch Distribution Modeling (PaDiM) using ResNet-18 feature extractor."""

    def __init__(
        self,
        d_reduced: int = 100,
        random_seed: int = 42,
        device: str = "cpu",
    ):
        super().__init__()
        self.d_reduced = d_reduced
        self.device = torch.device(device)

        # Load pretrained ResNet-18 backbone
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.eval()
        for param in backbone.parameters():
            param.requires_grad = False

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # 64 channels
        self.layer2 = backbone.layer2  # 128 channels
        self.layer3 = backbone.layer3  # 256 channels

        # Total channels = 64 + 128 + 256 = 448
        total_dim = 64 + 128 + 256

        # Set fixed random indices for dimension reduction
        rng = np.random.RandomState(random_seed)
        self.selected_indices = rng.choice(total_dim, self.d_reduced, replace=False)
        self.selected_indices = torch.tensor(self.selected_indices, dtype=torch.long)

        # Memory bank parameters (fitted on normal images)
        self.mean: torch.Tensor | None = None        # (d_reduced, H_p, W_p)
        self.inv_cov: torch.Tensor | None = None    # (H_p, W_p, d_reduced, d_reduced)

        self.to(self.device)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract multi-scale patch embeddings and apply random dimension reduction.

        Args:
            x: Input images tensor (B, 3, H, W)

        Returns:
            Reduced patch embeddings (B, d_reduced, H_p, W_p)
        """
        x = x.to(self.device)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            f1 = self.layer1(x)  # (B, 64, H/4, W/4)
            f2 = self.layer2(f1) # (B, 128, H/8, W/8)
            f3 = self.layer3(f2) # (B, 256, H/16, W/16)

            # Upsample f2 and f3 to layer1 resolution
            h_p, w_p = f1.shape[2], f1.shape[3]
            f2_up = F.interpolate(f2, size=(h_p, w_p), mode="bilinear", align_corners=False)
            f3_up = F.interpolate(f3, size=(h_p, w_p), mode="bilinear", align_corners=False)

            # Concatenate along channel dimension: (B, 448, H_p, W_p)
            features = torch.cat([f1, f2_up, f3_up], dim=1)

            # Subsample channels to d_reduced
            indices = self.selected_indices.to(self.device)
            features = torch.index_select(features, 1, indices)

        return features

    def fit(self, normal_dataloader: torch.utils.data.DataLoader) -> None:
        """Fit Multivariate Gaussian Distribution at each patch location on normal images.

        Args:
            normal_dataloader: DataLoader containing only defect-free (Golden Template) images.
        """
        self.eval()
        all_features: list[torch.Tensor] = []

        with torch.no_grad():
            for batch in normal_dataloader:
                if isinstance(batch, (tuple, list)):
                    imgs = batch[0]
                else:
                    imgs = batch
                feat = self.extract_features(imgs)
                all_features.append(feat.cpu())

        # Concatenate across all batches: (N, d, H_p, W_p)
        all_features_tensor = torch.cat(all_features, dim=0)
        n, d, h_p, w_p = all_features_tensor.shape

        # Reshape to (N, d, H_p * W_p)
        features_flat = all_features_tensor.view(n, d, h_p * w_p)

        # Compute mean at each patch location: (d, H_p * W_p)
        mean = torch.mean(features_flat, dim=0)

        # Compute covariance matrix at each patch location
        # (x - mean): (N, d, H_p * W_p)
        zero_mean = features_flat - mean.unsqueeze(0)

        # Covariance: (H_p * W_p, d, d)
        cov = torch.zeros((h_p * w_p, d, d), dtype=torch.float32)
        identity = torch.eye(d, dtype=torch.float32) * 0.01  # Regularization epsilon

        for p in range(h_p * w_p):
            patch_data = zero_mean[:, :, p]  # (N, d)
            c = (patch_data.T @ patch_data) / (n - 1) + identity
            cov[p] = c

        # Compute inverse covariance matrices
        inv_cov = torch.linalg.inv(cov)

        self.mean = mean.view(d, h_p, w_p).to(self.device)
        self.inv_cov = inv_cov.view(h_p, w_p, d, d).to(self.device)

    def predict(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Compute Mahalanobis distance anomaly maps and image-level anomaly scores.

        Args:
            x: Input image tensor (B, 3, H, W)

        Returns:
            image_scores: Array of image anomaly scores (B,)
            heatmaps: Array of anomaly heatmaps (B, H, W) in range [0, 1]
        """
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("PaDiM model must be fitted with normal images before predict()")

        self.eval()
        b, c, orig_h, orig_w = x.shape
        features = self.extract_features(x)  # (B, d, H_p, W_p)
        b, d, h_p, w_p = features.shape

        # Zero-mean features: (B, d, H_p, W_p)
        delta = features - self.mean.unsqueeze(0)

        # Reshape delta to (B, H_p, W_p, d)
        delta = delta.permute(0, 2, 3, 1)

        # Compute Mahalanobis distance at each patch (B, H_p, W_p)
        # dist_sq = delta @ inv_cov @ delta.T
        # delta.unsqueeze(-2): (B, H_p, W_p, 1, d)
        # inv_cov: (H_p, W_p, d, d)
        # matmul: (B, H_p, W_p, 1, d)
        temp = torch.matmul(delta.unsqueeze(-2), self.inv_cov.unsqueeze(0))
        dist_sq = torch.matmul(temp, delta.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))  # (B, H_p, W_p)

        # Upsample distance map to original image resolution
        dist_up = F.interpolate(
            dist.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)  # (B, orig_h, orig_w)

        heatmaps = dist_up.cpu().numpy()

        # Apply Gaussian blur for smooth anomaly localization and compute max score
        image_scores = []
        smooth_heatmaps = []

        for i in range(b):
            hm = heatmaps[i]
            hm_blurred = cv2.GaussianBlur(hm, (33, 33), sigmaX=4.0)
            score = float(np.max(hm_blurred))
            image_scores.append(score)
            smooth_heatmaps.append(hm_blurred)

        return np.array(image_scores), np.array(smooth_heatmaps)

    def save(self, filepath: Path) -> None:
        """Save fitted PaDiM parameters to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "d_reduced": self.d_reduced,
            "selected_indices": self.selected_indices,
            "mean": self.mean.cpu() if self.mean is not None else None,
            "inv_cov": self.inv_cov.cpu() if self.inv_cov is not None else None,
        }, filepath)

    def load(self, filepath: Path) -> None:
        """Load fitted PaDiM parameters from disk."""
        data = torch.load(filepath, map_location=self.device)
        self.d_reduced = data["d_reduced"]
        self.selected_indices = data["selected_indices"]
        if data["mean"] is not None:
            self.mean = data["mean"].to(self.device)
        if data["inv_cov"] is not None:
            self.inv_cov = data["inv_cov"].to(self.device)
