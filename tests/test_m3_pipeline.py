"""Unit tests for Milestone 3: COCO Dataset, Faster R-CNN model, and evaluation."""

from pathlib import Path

import pytest
import torch

from pcbdetect.data.coco_dataset import PCBCocoDataset, detection_collate_fn
from pcbdetect.models.faster_rcnn import build_faster_rcnn, unfreeze_backbone_top_layers


def test_coco_dataset_loading():
    """Verify PCBCocoDataset can load train.json annotations."""
    train_json = Path("data/processed/coco/annotations/instances_train.json")
    images_dir = Path("data/processed/coco/images/train")

    if not train_json.exists():
        pytest.skip("Processed COCO data not found")

    dataset = PCBCocoDataset(annotations_json=train_json, images_dir=images_dir)
    assert len(dataset) > 0, "Dataset should have samples"

    img_tensor, target = dataset[0]
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.dim() == 3  # (C, H, W)
    assert img_tensor.shape[0] == 3
    assert "boxes" in target
    assert "labels" in target
    assert "image_id" in target

    # Verify collate function
    batch = [dataset[0], dataset[1]]
    images, targets = detection_collate_fn(batch)
    assert len(images) == 2
    assert len(targets) == 2


def test_faster_rcnn_architecture():
    """Verify Faster R-CNN model builder and freeze/unfreeze logic."""
    model = build_faster_rcnn(num_classes=7, pretrained_backbone=False, freeze_backbone=True)
    assert model is not None

    # Check that backbone is frozen
    for p in model.backbone.parameters():
        assert not p.requires_grad

    # Unfreeze top layers
    unfreeze_backbone_top_layers(model, unfreeze_fpn=True, unfreeze_layer4=True)
    if hasattr(model.backbone, "fpn"):
        for p in model.backbone.fpn.parameters():
            assert p.requires_grad

    # Forward pass on dummy batch in eval mode
    model.eval()
    dummy_imgs = [torch.rand(3, 300, 300)]
    with torch.no_grad():
        preds = model(dummy_imgs)

    assert len(preds) == 1
    assert "boxes" in preds[0]
    assert "scores" in preds[0]
    assert "labels" in preds[0]
