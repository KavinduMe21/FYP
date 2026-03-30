"""
model.py — R3D-18 Transfer Learning Model for Knock-On Classification
======================================================================
Uses a pre-trained R3D-18 (ResNet 3D, 18 layers) from torchvision.
The model was originally trained on the Kinetics-400 video dataset
(400 human action classes), so it already understands motion patterns.

We freeze the early convolutional layers and only fine-tune the later
layers + a new classification head for our 2 classes:
    0 = normal_play
    1 = knock_on

Input shape : (batch, 3, 16, 224, 224)  —  3 RGB channels, 16 frames
Output shape: (batch, 2)                 —  class logits
"""

import torch
import torch.nn as nn
import torchvision.models.video as video_models


def create_model(num_classes=2, pretrained=True, freeze_early=True):
    """
    Build an R3D-18 model for binary video classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (2 for knock_on vs normal_play).
    pretrained : bool
        If True, load Kinetics-400 pre-trained weights.
    freeze_early : bool
        If True, freeze stem + layer1 + layer2 so only layer3, layer4,
        and the classifier head are trained.  This speeds up training
        and prevents overfitting on a small dataset.

    Returns
    -------
    model : nn.Module
    """
    # Load the pre-trained R3D-18
    if pretrained:
        weights = video_models.R3D_18_Weights.DEFAULT
        model = video_models.r3d_18(weights=weights)
        print("[Model] Loaded R3D-18 with Kinetics-400 pre-trained weights")
    else:
        model = video_models.r3d_18(weights=None)
        print("[Model] Loaded R3D-18 (random init, no pre-trained weights)")

    # Freeze early layers to avoid overfitting on small dataset
    if freeze_early:
        frozen_count = 0
        for name, param in model.named_parameters():
            # Keep layer3, layer4, and fc trainable
            if not any(part in name for part in ["layer3", "layer4", "fc"]):
                param.requires_grad = False
                frozen_count += 1
        print(f"[Model] Frozen {frozen_count} early-layer parameter groups")

    # Replace the final fully-connected layer
    # Original: nn.Linear(512, 400)  →  ours: Dropout + nn.Linear(512, 2)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )

    # Count parameters
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Parameters: {total:,} total, {trainable:,} trainable")

    return model


# ═══════════════════════════════════════════════════════════════════
#  Quick self-test
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model = create_model(num_classes=2, pretrained=True, freeze_early=True)

    # Test with a dummy 16-frame clip batch
    dummy_input = torch.randn(2, 3, 16, 224, 224)   # batch of 2 clips
    output = model(dummy_input)
    print(f"\nInput  shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")           # (2, 2)
    print(f"Output      : {output}")
