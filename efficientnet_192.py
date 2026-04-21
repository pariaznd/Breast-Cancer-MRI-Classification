import torch
from monai.networks.nets import EfficientNetBN

def get_model(num_classes=3):
    model = EfficientNetBN(
        model_name="efficientnet-b4",
        spatial_dims=3,
        in_channels=5,
        num_classes=num_classes,
        pretrained=False
    )
    return model
