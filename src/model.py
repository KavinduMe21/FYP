import torch
import torch.nn as nn
import torchvision.models.video as video_models


def create_model(num_classes=2, pretrained=True, freeze_early=True):
    # R3D-18 pretrained on Kinetics-400
    if pretrained:
        weights = video_models.R3D_18_Weights.DEFAULT
        model = video_models.r3d_18(weights=weights)
        print("Loaded R3D-18 with Kinetics-400 pretrained weights")
    else:
        model = video_models.r3d_18(weights=None)
        print("Loaded R3D-18 from scratch (no pretrained weights)")

    # freeze stem + layer1/2 to avoid overfitting
    if freeze_early:
        frozen = 0
        for name, param in model.named_parameters():
            if not any(part in name for part in ["layer3", "layer4", "fc"]):
                param.requires_grad = False
                frozen += 1
        print(f"Froze {frozen} early parameter groups")
    
    # replace fc head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total, {trainable:,} trainable")

    return model


if __name__ == "__main__":
    model = create_model(num_classes=2, pretrained=True, freeze_early=True)

    # sanity check
    dummy = torch.randn(2, 3, 16, 224, 224)
    out = model(dummy)
    print(f"\nInput shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output: {out}")
