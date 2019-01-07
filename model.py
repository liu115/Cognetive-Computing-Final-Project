import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import numpy as np
import os

class AutoEncoder(nn.Module):
    def __init__(self, fc_dims=256, face_size=(128, 128), target_size=(128, 128), output_size=(128, 128)):
        super(AutoEncoder, self).__init__()
        self.fc_dims = fc_dims
        self.face_size = face_size
        self.target_size = target_size
        self.output_size = output_size

        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

        self.face_fc = nn.Sequential(
            nn.Linear(self.face_size[0] * self.face_size[1] // 256 * 256, self.fc_dims)
        )

        self.target_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
        )

        self.target_fc = nn.Sequential(
            nn.Linear(self.target_size[0] * self.target_size[1] // 256 * 256, self.fc_dims)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(True),
            nn.Sigmoid()
        )

        self.output_fc = nn.Sequential(
            nn.Linear(self.fc_dims * 2, self.output_size[0] * self.output_size[1] // 256 * 256)
        )

    def forward(self, face, target):
        face = self.face_encoder(face)
        face = face.view(face.size(0), -1)
        face = self.face_fc(face)

        target = self.target_encoder(target)
        target = target.view(target.size(0), -1)
        target = self.target_fc(target)
        
        x = torch.cat([face, target], 1)
        x = self.output_fc(x)
        x = x.view(x.size(0), -1, self.output_size[0] // 16, self.output_size[1] // 16)
        x = self.decoder(x)
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
