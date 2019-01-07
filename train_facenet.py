import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import os
import time
import random

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
test_data = CelebaDataset(
    '/tmp2/liu115/Anno/identity_CelebA.txt',
    '/tmp2/liu115/img_align_celeba',
    custom_transform,
    training=False
)

train_loader = DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=1,
    shuffle=True,
    num_workers=1
)


model = FaceNet().cuda()
model.train()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
#criterion = torch.nn.MSELoss()


def test():
    start_time = time.time()
    embeddings = []
    for batch_idx, imgs in enumerate(test_loader):
        imgs = torch.cat(imgs, dim=0)
        imgs = imgs.cuda()
        output = model(imgs)
        embeddings.append(output.detach().cpu())
    
    # Calc average intra-class distance
    intra_class_dist = 0
    num_pairs = 0
    for idx, embedding in enumerate(embeddings):
        num_imgs = embedding.shape[0]
        for i in range(num_imgs):
            for j in range(i+1, num_imgs):
                intra_class_dist += np.linalg.norm(embedding[i] - embedding[j])
                num_pairs += 1
    intra_class_dist /= (1. * num_pairs)
    
    # Calc average inter-class distance
    inter_class_dist = 0
    num_pairs = 0
    num_samples = 20
    sample_embeddings = random.sample(embeddings, num_samples)

    for i in range(num_samples):
        for j in range(i+1, num_samples):
            ix = random.randrange(sample_embeddings[i].shape[0])
            jx = random.randrange(sample_embeddings[j].shape[0])

            inter_class_dist += np.linalg.norm(sample_embeddings[i][ix] - sample_embeddings[j][jx])
            num_pairs += 1
    inter_class_dist /= (1. * num_pairs)
    
    took = time.time() - start_time
    print('intra:{:.5f}, inter:{:.5f}, took:{:.1f}s'.format(intra_class_dist, inter_class_dist, took))

def train():
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
        test()
        
    #for i in range (10):
    #    img = output[i].permute(1, 2, 0)
    #    plt.imshow(img.detach().cpu())
    #    plt.savefig('output/{}_{}.png'.format(epoch, i))
train()
