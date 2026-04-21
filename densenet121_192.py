import torch
import torch.nn as nn
from monai.networks.nets import DenseNet121

def get_model(num_classes=3):
    model = DenseNet121(
        spatial_dims=3,
        in_channels=5,
        out_channels=num_classes
    )
    return model

if __name__ == "__main__":
    model = get_model()
    x = torch.randn(1, 5, 256, 256, 32)
    out = model(x)
    print("Output shape:", out.shape)