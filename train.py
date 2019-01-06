import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import os

from ae import AutoEncoder
from data import FruitDataset

LEARNING_RATE = 1e-4
num_epochs = 100
BATCH_SIZE = 128
OUTPUT_DIR = './output'

custom_transform = transforms.Compose([transforms.ToTensor()])
train_data = FruitDataset('/tmp2/liu115/fruits-360/Training', transform=custom_transform)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)


model = AutoEncoder().cuda()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
criterion = torch.nn.MSELoss()

if not os.path.exists(OUTPUT_DIR):
    os.mkdirs(OUTPUT_DIR)

for epoch in range(num_epochs):
    for batch_idx, img in enumerate(train_loader):
        
        img = img.cuda()
        output = model(img)
        loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch, num_epochs, loss.item()))
    for i in range (10):
        img = output[i].permute(1, 2, 0)
        plt.imshow(img.detach().cpu())
        plt.savefig('output/{}_{}.png'.format(epoch, i))

