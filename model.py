import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np
import os

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        self.target_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, target):
        # face = self.face_encoder(face)
        target = self.target_encoder(target)
        # x = torch.cat([face, target], 1)
        x = self.decoder(target)
        return x

class FaceNet(nn.Module):
    def __init__(self):
        super(FaceNet, self).__init__()

        self.fc = nn.Linear(512 * 42, 512)
        self.model = torchvision.models.resnet18()
    
    def l2_norm(self, x):
        input_size = x.size()
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(x, norm.view(-1, 1).expand_as(x))
        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        self.features = self.l2_norm(x) * 10
        return self.features


def l2_dist(x1, x2):
    assert x1.size() == x2.size()
    eps = 1e-4 / x1.size(1)
    diff = torch.abs(x1 - x2)
    out = torch.pow(diff, 2).sum(dim=1)
    return torch.pow(out + eps, 0.5)


def triplet_loss(anchor, positive, negative):
    # Set margin = 0.5
    d_p = l2_dist(anchor, positive)
    d_n = l2_dist(anchor, negative)

    dist_hinge = torch.clamp(0.5 + d_p - d_n, min=0.0)
    loss = torch.mean(dist_hinge)
    return loss
