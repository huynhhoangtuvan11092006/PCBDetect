"""Faster R-CNN model builder with ResNet-50-FPN backbone for PCB defect detection.

Supports:
- 7 classes: 6 defect categories (open, short, mousebite, spur, copper, pin_hole) + 1 background.
- Backbone freezing / unfreezing stages for controlled transfer learning.
"""


from torchvision.models.detection import (
    FasterRCNN,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_faster_rcnn(
    num_classes: int = 7,  # 6 defect classes + 1 background
    pretrained_backbone: bool = True,
    freeze_backbone: bool = False,
    trainable_backbone_layers: int | None = None,
) -> FasterRCNN:
    """Construct Faster R-CNN model with ResNet-50 + FPN backbone.

    Args:
        num_classes: Total number of classes including background (default: 7).
        pretrained_backbone: If True, initializes weights from COCO pretrained model.
        freeze_backbone: If True, disables gradient updates for the entire ResNet-50 backbone.
        trainable_backbone_layers: Number of trainable ResNet layers from 0 to 5.

    Returns:
        torchvision FasterRCNN instance configured for PCB detection.
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained_backbone else None

    # By default, trainable_backbone_layers=3 in torchvision (layers 3, 4, 5 trainable)
    if trainable_backbone_layers is not None:
        model = fasterrcnn_resnet50_fpn(
            weights=weights,
            trainable_backbone_layers=trainable_backbone_layers,
        )
    else:
        model = fasterrcnn_resnet50_fpn(weights=weights)

    # Replace the classification head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model


def unfreeze_backbone_top_layers(
    model: FasterRCNN,
    unfreeze_fpn: bool = True,
    unfreeze_layer4: bool = True,
) -> None:
    """Unfreeze top ResNet layers and FPN feature fusion for fine-tuning stage 2."""
    if unfreeze_fpn and hasattr(model.backbone, "fpn"):
        for param in model.backbone.fpn.parameters():
            param.requires_grad = True

    if unfreeze_layer4 and hasattr(model.backbone, "body") and hasattr(model.backbone.body, "layer4"):
        for param in model.backbone.body.layer4.parameters():
            param.requires_grad = True
