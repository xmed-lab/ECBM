import torch
import torch.nn as nn
import numpy as np

from torchvision import transforms


class ResNetBottom(nn.Module):
    def __init__(self, original_model):
        super(ResNetBottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        return x


class ResNetTop(nn.Module):
    def __init__(self, original_model):
        super(ResNetTop, self).__init__()
        self.features = nn.Sequential(*[list(original_model.children())[-1]])

    def forward(self, x):
        x = self.features(x)
        x = nn.Softmax(dim=-1)(x)
        return x



def get_model():
    from torchvision.models import resnet101, ResNet101_Weights
    model=resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
    backbone, model_top = ResNetBottom(model), ResNetTop(model)
    return backbone