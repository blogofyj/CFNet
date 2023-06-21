import torch, os, sys, torchvision, argparse
device = torch.device('cuda')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
from torch import nn
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from metrics import psnr, ssim
from purifier_network import Dehaze
from CR import *
from torch.utils import data
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.transforms as transforms
from resnet_model import ResNet18
import json

warnings.filterwarnings('ignore')

start_time = time.time()

classifier = ResNet18()
classifier.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/DC-VAE-main/models/Result/ResNet18/params_finished.pt'))
classifier = classifier.cuda()

class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]

class Dataset(data.Dataset):
    def __init__(self, data, labels, adv_data):
        self.data = torch.from_numpy(data)
        self.labels = torch.from_numpy(labels)
        self.adv_data = torch.from_numpy(adv_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X = self.data[index]
        y = self.labels[index]
        z = self.adv_data[index]

        return X, y, z


data = np.load('/remote-home/cs_igps_yangjin/cifar10_data/xs_cifar10.npy')  # image data in npy file
labels = np.load('/remote-home/cs_igps_yangjin/cifar10_data/ys_cifar10.npy')  # labels data in npy file
adv_data = np.load('/remote-home/cs_igps_yangjin/cifar10_data/advs_cifar10.npy')  # adversarial image data in npy file
dataset = Dataset(data, labels, adv_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=6)

transform = transforms.Compose([transforms.ToTensor()])
test_data = torchvision.datasets.CIFAR10(root='./cifar10', train=False, transform=transform, download=False)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=32, shuffle=True)

# train_data = torchvision.datasets.CIFAR10(
#     root='./cifar10',  # 下载路径
#     train=True,  # 下载训练数据
#     transform=transform,  # 将数据转化为tensor类型
#     download=True  # 是否下载MNIST数据集
# )
# test_data = torchvision.datasets.CIFAR10(
#     root='./cifar10',
#     train=False,
#     download = True,
#     transform = transform
# )
# # 获得测试数据集
#
# # 将dataset放入DataLoader中  (batch, channel, 28, 28)
# trainloader = data.DataLoader(
#     dataset=train_data,
#     batch_size=64,  # 设置batch size
#     shuffle=True  # 打乱数据
# )
# testloader = data.DataLoader(
#     dataset=test_data,
#     batch_size=64,  # 设置batch size
#     shuffle=True  # 打乱数据
# )

def lr_schedule_cosdecay(t, T, init_lr=0.0001):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def train(net, loader_train, loader_test, optim, criterion, scheduler):
    losses = []
    rec_losses = []
    vgg_losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []

    for epoch in range(1, epochs + 1):
        for batch_idx, (images, labels, adv_data) in enumerate(loader_train):
            # step = batch_idx
            # steps = 500000
            image = images.cuda()
            adv_data = adv_data.cuda()
            labels = labels.cuda()
            optim.zero_grad()

            out = net(adv_data)

            loss_vgg7, all_ap, all_an, loss_rec = 0, 0, 0, 0

            loss_rec = criterion[0](out, image)

            loss_vgg7, all_ap, all_an = criterion[1](out, image, adv_data)

            loss = loss_rec*10 + loss_vgg7
            loss.backward()

            optim.step()
            losses.append(loss.item())
            rec_losses.append(loss_rec.item())
            vgg_losses.append(loss_vgg7.item())

        scheduler.step()

        print("epoch {}'s: loss:{:.5f}, rec_losses:{:.5f}, vgg_losses:{:.5f}".format(epoch, np.sum(losses) / len(losses), \
                                                                                     np.sum(rec_losses) / len(rec_losses), \
                                                                                     np.sum(vgg_losses) / len(vgg_losses)))

        if epoch % 5 == 0:
            torch.save(net.state_dict(), '{}/params_{}.pt'.format(model_dir, epoch))
            # torch.save(Dis.state_dict(), '{}/params_{}.pt'.format(Dis_dir, epoch))

        torch.save(net.state_dict(), '{}/params_pause.pt'.format(model_dir))
        print("Current epoch saved!")

        if epoch % 3 == 0:
            test(epoch, net, loader_test)


    torch.save(net.state_dict(), '{}/params_finished.pt'.format(model_dir))


def test(epoch, net, loader_test):
    net.eval()
    total = 0
    correct = 0

    for i, (images, labels) in enumerate(loader_test):
        inputs = images.cuda()
        y = labels.cuda()
        with torch.no_grad():
            pred = net(inputs)
            logit = classifier(pred)
            prediction = torch.max(logit, 1)[1]
            correct = correct + torch.eq(prediction, y).float().sum().item()

            total = total + inputs.size(0)
    accuracy = correct / total
    print()
    print('TEST *TOP* ACC:{:.4f} at e:{:03d}'.format(accuracy, epoch))
    print()




def set_seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    set_seed_torch(666)

    model_dir = './purifier_network3'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_name = 'test2'
    epochs = 100
    lr = 0.0001 #(0.0001, 0.001)
    loader_train = dataloader
    net = Dehaze(3, 3)
    net = net.cuda()

    criterion = []
    # criterion.append(nn.L1Loss().cuda())
    criterion.append(nn.MSELoss().cuda())
    criterion.append(ContrastLoss(ablation=False))

    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=lr, betas=(0.9, 0.999),
                           eps=1e-08)
    scheduler = MinExponentialLR(optimizer, gamma=0.998, minimum=1e-5)

    optimizer.zero_grad()
    # train(net, trainloader, testloader, optimizer, criterion)
    train(net, dataloader, testloader, optimizer, criterion, scheduler)

    # net.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/purifier_vae/purifier_network.best["model"]'))
