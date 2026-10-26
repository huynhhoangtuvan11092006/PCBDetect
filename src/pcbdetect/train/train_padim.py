"""Training/Fitting script for PaDiM on normal PCB golden templates with MLflow tracking.

PaDiM is trained solely on defect-free (Golden Template) PCB images.
Fits multivariate Gaussian distributions over patch embeddings and saves weights.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mlflow
import torch
from torch.utils.data import DataLoader, Dataset

from pcbdetect.data.split import create_group_disjoint_split
from pcbdetect.models.padim import PaDiM

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class PaDiMConfig:
    d_reduced: int = 100
    img_size: int = 256  # 256x256 enables fast CPU extraction while preserving trace geometry
    batch_size: int = 16
    max_train_samples: int = 120
    output_model_path: Path = Path("models/padim_resnet18.pt")
    experiment_name: str = "PCBDetect-Anomaly-PaDiM"
    run_name: str = "padim_resnet18_normal_only"
    device: str = "cpu"


class NormalPCBDataset(Dataset):
    """Dataset loading only pristine golden template PCB images."""

    def __init__(self, image_paths: list[Path], img_size: int = 256):
        self.image_paths = image_paths
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise FileNotFoundError(f"Failed to read template image at {img_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if (rgb.shape[0], rgb.shape[1]) != (self.img_size, self.img_size):
            rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        # Normalize with ImageNet mean and std
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor


def train_padim(cfg: PaDiMConfig) -> Path:
    """Fit PaDiM model on normal golden templates and track with MLflow."""
    logger.info("Loading training split to extract normal golden templates...")
    splits = create_group_disjoint_split()
    train_samples = splits.train

    # Collect template image paths
    template_paths: list[Path] = []
    seen = set()
    for s in train_samples:
        p = s.template_image_path
        if p.exists() and str(p) not in seen:
            seen.add(str(p))
            template_paths.append(p)

    if cfg.max_train_samples and len(template_paths) > cfg.max_train_samples:
        template_paths = template_paths[:cfg.max_train_samples]

    logger.info(f"Fitting PaDiM on {len(template_paths)} pristine normal golden template images.")

    dataset = NormalPCBDataset(template_paths, img_size=cfg.img_size)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)

    model = PaDiM(d_reduced=cfg.d_reduced, device=cfg.device)

    mlflow.set_experiment(cfg.experiment_name)
    with mlflow.start_run(run_name=cfg.run_name) as run:
        mlflow.log_params({
            "model_type": "PaDiM",
            "backbone": "ResNet-18",
            "d_reduced": cfg.d_reduced,
            "img_size": cfg.img_size,
            "num_normal_train_samples": len(template_paths),
            "labeling_cost": "0 annotations (Unsupervised Normal-Only)",
        })

        t0 = time.time()
        logger.info("Extracting multi-scale patch embeddings and calculating Gaussian distributions...")
        model.fit(loader)
        duration = time.time() - t0
        logger.info(f"PaDiM fitted successfully in {duration:.2f} seconds!")

        mlflow.log_metric("fit_time_seconds", duration)

        model.save(cfg.output_model_path)
        logger.info(f"Saved PaDiM model checkpoint to {cfg.output_model_path}")
        mlflow.log_artifact(str(cfg.output_model_path))

    return cfg.output_model_path


if __name__ == "__main__":
    train_padim(PaDiMConfig())
