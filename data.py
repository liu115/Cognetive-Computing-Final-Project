import pandas as pd
import numpy as np
import random
import os

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

# txt_path = '/tmp2/liu115/Anno/identity_CelebA.txt'
# df = pd.read_csv(txt_path, sep=' ', names=['path', 'id'], dtype={'path': str, 'id': np.int32})
# print(df['path'].values)

class FruitDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.transform = transform
        fruit_dirs = os.listdir(img_dir)
        self.img_paths = []
        for fruit_dir in fruit_dirs:
            for img_name in os.listdir(os.path.join(img_dir, fruit_dir)):
                self.img_paths.append(os.path.join(img_dir, fruit_dir, img_name))
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        img = img.resize((128, 128), Image.BILINEAR)
        if self.transform:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.img_paths)

def sep_celeba_train_val():
    txt_path = '/tmp2/liu115/Anno/identity_CelebA.txt'
    df = pd.read_csv(txt_path, sep=' ', names=['path', 'id'], dtype={'path': str, 'id': np.int32})

    num_train = int(df.shape[0] * 0.8)
    df_train = df.iloc[:num_train, :]
    df_test = df.iloc[num_train:, :]

    def_train.to_csv('/tmp2/liu115/Anno/identity_CelebA_train.txt', sep=" ")

class CelebaDataset(Dataset):
    def __init__(self, txt_path, img_dir, transform=None, training=True):
        df = pd.read_csv(txt_path, sep=' ', names=['path', 'id'], dtype={'path': str, 'id': np.int32})

        self.img_dir = img_dir
        self.txt_path = txt_path
        self.transform = transform
        self.img_ids = df['id'].values
        self.img_names = df['path'].values
        self.training = training

        self.id_dict = {}
        for i, img_id in enumerate(self.img_ids):
            if img_id in self.id_dict:
                self.id_dict[img_id].append(self.img_names[i])
            else:
                self.id_dict[img_id] = [self.img_names[i]]
        
        del_id_list = []
        for id, imgs in self.id_dict.items():
            if len(imgs) < 2:
                del_id_list.append(id)
        for id in del_id_list:
            del self.id_dict[id]
        assert len(self.id_dict) > 2

        self.id_dict = list(self.id_dict.values())
        
        num_training = len(self.id_dict) * 8 // 10
        if self.training:
            self.id_dict = self.id_dict[:num_training]
        else:
            self.id_dict = self.id_dict[num_training:]

    def read_img(self, name):
        img = Image.open(os.path.join(self.img_dir, name))
        if self.transform:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        # Sample anchor, positive, and negative samples
        
        if self.training:
            sample_size = len(self.id_dict[index])
            # assert sample_size > 1
            anchor, positive = random.choices(self.id_dict[index], k=2)

            neg_index = random.randrange(self.__len__())
            while neg_index == index:
                neg_index = random.randrange(self.__len__())
            negative = random.choice(self.id_dict[neg_index])
            
            return self.read_img(anchor), self.read_img(positive), self.read_img(negative)
        else:
            return [self.read_img(img) for img in self.id_dict[index]]

    def __len__(self):
        return len(self.id_dict)

custom_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = CelebaDataset(
    '/tmp2/liu115/Anno/identity_CelebA.txt',
    '/tmp2/liu115/img_align_celeba',
    custom_transform,
    training=True
)
test_dataset = CelebaDataset(
    '/tmp2/liu115/Anno/identity_CelebA.txt',
    '/tmp2/liu115/img_align_celeba',
    custom_transform,
    training=False
)
print(len(train_dataset))
print(len(test_dataset))

# a = test_dataset.__getitem__(0)
# a = a.permute(1, 2, 0)
# b = b.permute(1, 2, 0)
# c = c.permute(1, 2, 0)


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4
)

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.manual_seed(0)
# num_epochs = 2
# for epoch in range(num_epochs):
# 
#     for batch_idx, (x, y) in enumerate(train_loader):
#         
#         print('Epoch:', epoch+1, end='')
#         print(' | Batch index:', batch_idx, end='')
#         print(' | Batch size:', y.size()[0])
#         
#         x = x.to(device)
#         y = y.to(device)
#         print(x.shape)
#         one_img = x[0].permute(1, 2, 0)
#         print(one_img.shape)
#         plt.imshow(one_img.cpu())
#         plt.savefig('test.png')
#         break
# fd = FruitDataset('/tmp2/liu115/fruits-360/Training')
# print(len(fd))
# print(fd[10])
