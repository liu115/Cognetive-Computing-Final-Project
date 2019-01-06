import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import os
import time

from model import AutoEncoder, FaceNet, triplet_loss
from data import CelebaDataset

LEARNING_RATE = 1e-4
num_epochs = 100
BATCH_SIZE = 64
OUTPUT_DIR = './facenet_output'

custom_transform = transforms.Compose([transforms.ToTensor()])
train_data = CelebaDataset(
    '/tmp2/liu115/Anno/identity_CelebA.txt',
    '/tmp2/liu115/img_align_celeba',
    custom_transform
)

train_loader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)


model = FaceNet().cuda()
model.train()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
#criterion = torch.nn.MSELoss()

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

start_time = time.time()
for epoch in range(num_epochs):
    for batch_idx, (anchor_img, pos_img, neg_img) in enumerate(train_loader):
        
        anchor_img = anchor_img.cuda()
        pos_img = pos_img.cuda()
        neg_img = neg_img.cuda()

        anchor_output = model(anchor_img)
        pos_output = model(pos_img)
        neg_output = model(neg_img)
        loss = triplet_loss(anchor_output, pos_output, neg_output)
        # loss = criterion(output, img)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 50 == 0:
            took = time.time() - start_time
            print('epoch [{}/{}], loss:{:.4f}, took: {:.1f}s'.format(epoch, num_epochs, loss.item(), took))
            start_time = time.time()

    torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    },
        os.path.join(OUTPUT_DIR, 'checkpoint_{}.pth'.format(epoch))
    )
    
    #for i in range (10):
    #    img = output[i].permute(1, 2, 0)
    #    plt.imshow(img.detach().cpu())
    #    plt.savefig('output/{}_{}.png'.format(epoch, i))

