import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class FishNet(nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        base_model = mobilenet_v3_small(pretrained=True)
        self.features = base_model.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)