import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import os, time

from model import AutoEncoder
from data import FruitDataset

LEARNING_RATE = 1e-4
num_epochs = 100
BATCH_SIZE = 32
OUTPUT_DIR = './output'

custom_transform = transforms.Compose([transforms.ToTensor()])
train_data = FruitDataset('/tmp2/liu115/fruits-360/Training', transform=custom_transform)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False,
    num_workers=2
)


model = AutoEncoder().cuda()
model.train()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
criterion = torch.nn.MSELoss()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

start_time = time.time()
for epoch in range(num_epochs):
    for batch_idx, img in enumerate(train_loader):
        
        img = img.cuda()
        output = model(img, img)
        loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 50 == 0:
            took = time.time() - start_time
            print('epoch [{}/{}], loss:{:.4f}, took: {:.1f}s'.format(epoch, num_epochs, loss.item(), took))
            start_time = time.time()
            for i in range(10):
                img = output[i].permute(1, 2, 0)
                plt.imshow(img.detach().cpu())
                plt.savefig('output/{}_{}.png'.format(epoch, i))

    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    },
        os.path.join(OUTPUT_DIR, 'checkpoint_{}.pth'.format(epoch))
    )
    
    print('epoch [{}/{}], loss:{:.4f}'.format(epoch, num_epochs, loss.item()))

