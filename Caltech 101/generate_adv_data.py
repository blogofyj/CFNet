import sys, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
sys.path.insert(0, os.path.abspath('..'))
import warnings

warnings.filterwarnings("ignore")
import foolbox
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST, FashionMNIST
from test.attacks import *
from advertorch.attacks import *
from purifier_vae.adversarial_one import *
from utils.classifier import *
from new import train_dataset
from torchvision.models import alexnet


# argument parser


if __name__ == "__main__":
    # get arguments
    # init and load model
    net = alexnet()
    net.classifier[6] = nn.Linear(4096, 101)
    net.load_state_dict(torch.load('./caltec_model/params_29.pt'))
    net.eval()
    net = net.cuda()

    # init dataset
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True)
    # adversarial methods
    # adv_list = ['fgsm']
    adv_list = ['fgsm', 'pgd_n']
    # test for accuracy
    xs = list()
    ys = list()
    advs = list()
    for image, label in train_dataloader:
        image = image.cuda()
        label = label.cuda()
        # batch += 1
        # print(batch)
        for i in range(1):
            for adv in adv_list:
                output, adv_out = add_adv(net, image, label, adv, i)
                output = net(output)
                adv_class = net(adv_out)
                print('attack method {}'.format(adv))
                print('actual class ', torch.argmax(output, 1))
                print('adversarial class ', torch.argmax(adv_class, 1))
                print('====================================')
                xs.append(image.cpu().detach().numpy())
                ys.append(label.cpu().detach().numpy())
                advs.append(adv_out.cpu().detach().numpy())

    adv_x = np.concatenate(advs, axis=0)
    xt = np.concatenate(xs, axis=0)
    yt = np.concatenate(ys, axis=0)

    if not os.path.exists('./Cal_data0.08'):
        os.mkdir('./Cal_data0.08')
    np.save('./Cal_data0.08/' + 'advs_caltech101.npy', adv_x)
    np.save('./Cal_data0.08/' + 'xs_caltech101.npy', xt)
    np.save('./Cal_data0.08/' + 'ys_caltech101.npy', yt)
