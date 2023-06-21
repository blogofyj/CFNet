from os import path
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt


def build_data(data_set,batch_size=20):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), #把灰度范围从0-255变换到0-1之间
        #transforms.Normalize(RGB_mean, RGB_std) #image=(image-mean)/std
    ])

    data_dir = path.join('./101_Categories', data_set)
    dataset = torchvision.datasets.ImageFolder(data_dir, transform=data_transform)
    dataloadder = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloadder


import os
from os import path
import numpy as np
from sklearn.model_selection import train_test_split

root = './101_Categories'
categories = os.listdir(root)

for i in range(len(categories)):
    category = categories[i]
    print(category)
    cat_dir = path.join(root, category)

    images = os.listdir(cat_dir)

    images, images_test = train_test_split(images, test_size=0.15)
    images_train, images_val = train_test_split(images, test_size=0.2)  # 训练集：验证集：测试集=14:3:3
    image_sets = images_train, images_test, images_val
    labels = 'train', 'test', 'val'

    for image_set, label in zip(image_sets, labels):
        dst_folder = path.join(root, label, category)  # 创建文件夹
        os.makedirs(dst_folder)
        os.rename(src_dir, dst_dir)

os.rmdir(cat_dir)  # 去除空文件夹
print(cnt1, cnt2, cnt3)