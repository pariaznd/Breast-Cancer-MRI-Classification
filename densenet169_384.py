import torch
from monai.networks.nets import DenseNet169

def get_model(num_classes=3):
    model = DenseNet169(
        spatial_dims=3,
        in_channels=5,
        out_channels=num_classes
    )
    return model

if __name__ == "__main__":
    model = get_model()
    x = torch.randn(1, 5, 128, 128, 32)
    out = model(x)
    print("Output shape:", out.shape)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total/1e6:.1f}M")
