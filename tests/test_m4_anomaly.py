"""Unit tests for Milestone 4 (M4) Anomaly Detection (PaDiM)."""

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from pcbdetect.models.padim import PaDiM


def test_padim_feature_extraction():
    """Verify PaDiM extracts multi-scale patch embeddings and reduces dimensions."""
    padim = PaDiM(d_reduced=100, device="cpu")
    dummy_input = torch.rand(2, 3, 128, 128)

    features = padim.extract_features(dummy_input)
    # Layer 1 downsamples by 4: 128 / 4 = 32
    assert features.shape == (2, 100, 32, 32)


def test_padim_fit_and_predict():
    """Verify PaDiM can fit on synthetic normal images and produce anomaly maps."""
    padim = PaDiM(d_reduced=50, device="cpu")

    # Synthetic normal images (low variance)
    normal_data = torch.ones(10, 3, 64, 64) * 0.5 + torch.randn(10, 3, 64, 64) * 0.01
    dataset = TensorDataset(normal_data)
    loader = DataLoader(dataset, batch_size=5)

    padim.fit(loader)
    assert padim.mean is not None
    assert padim.inv_cov is not None

    # Normal sample prediction
    test_normal = torch.ones(1, 3, 64, 64) * 0.5
    scores_normal, heatmaps_normal = padim.predict(test_normal)
    assert scores_normal.shape == (1,)
    assert heatmaps_normal.shape == (1, 64, 64)

    # Anomalous sample prediction (bright anomaly spot)
    test_anom = test_normal.clone()
    test_anom[0, :, 25:35, 25:35] = 1.0
    scores_anom, heatmaps_anom = padim.predict(test_anom)

    # Anomaly score should be higher on anomalous sample
    assert scores_anom[0] > scores_normal[0]


def test_padim_save_and_load():
    """Verify saving and reloading PaDiM model state."""
    padim = PaDiM(d_reduced=50, device="cpu")
    normal_data = torch.rand(6, 3, 64, 64)
    loader = DataLoader(TensorDataset(normal_data), batch_size=3)
    padim.fit(loader)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "padim_test.pt"
        padim.save(tmp_path)
        assert tmp_path.exists()

        padim_loaded = PaDiM(d_reduced=50, device="cpu")
        padim_loaded.load(tmp_path)
        assert padim_loaded.mean is not None
        assert padim_loaded.inv_cov is not None
        assert torch.allclose(padim.mean, padim_loaded.mean)
