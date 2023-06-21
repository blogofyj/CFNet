"""
no sigmoid, by jinyang
"""
import torch, os, sys, torchvision, argparse
device = torch.device('cuda')
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch, warnings
# from model_three import Dehaze
from mnist_model_two import Dehaze


from CR_mnist import *
from torch.utils import data
from torch.optim.lr_scheduler import ExponentialLR
import torchvision.transforms as transforms
from classifier import Classifier
import torch.fft as fft
from VGG_loss_mnist import VGGLoss
from matplotlib import pyplot as plt
import json

vggloss = VGGLoss(3,1, False).cuda()

warnings.filterwarnings('ignore')

start_time = time.time()

classifier = Classifier(28, 1)
classifier.load_state_dict(torch.load('classifier_mnist.pt'))
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


data = np.load('/remote-home/cs_igps_yangjin/adversarial dataset/xs_mnist_v2.npy')  # image data in npy file
labels = np.load('/remote-home/cs_igps_yangjin/adversarial dataset/ys_mnist_v2.npy')  # labels data in npy file
adv_data = np.load('/remote-home/cs_igps_yangjin/adversarial dataset/advs_mnist_v2.npy')  # adversarial image data in npy file
dataset = Dataset(data, labels, adv_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=6)

# transform = transforms.Compose([transforms.CenterCrop(28), transforms.ToTensor()])
transform = transforms.Compose([transforms.ToTensor()])
test_data = torchvision.datasets.MNIST(root='./MNIST_Test_Data', train=False, transform=transform, download=True)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=128, shuffle=True)



h, w = 28,28
lpf = torch.zeros((h, w))
R = (h+w)//8  #或其他
for x in range(w):
    for y in range(h):
        if ((x-(w-1)/2)**2 + (y-(h-1)/2)**2) < (R**2):
            lpf[y,x] = 1
hpf = 1-lpf
hpf, lpf = hpf.cuda(), lpf.cuda()

def pinglvyu(data):
    f = fft.fftn(data, dim=(2,3)).cuda()
    f = torch.roll(f, (h // 2, w // 2), dims=(2, 3))  # 移频操作,把低频放到中央
    f_l = f * lpf
    f_h = f * hpf
    X_l = torch.abs(fft.ifftn(f_l, dim=(2, 3)))
    X_h = torch.abs(fft.ifftn(f_h, dim=(2, 3)))
    return X_l, X_h

def lr_schedule_cosdecay(t, T, init_lr=0.0001):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def train(net, loader_train, loader_test, optim, criterion, scheduler):
    losses = []
    rec_losses = []
    vgg_losses = []
    pinglv_losses = []
    pertual_losses = []
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


            out, final_out = net(adv_data)
            img_h = out[:,0,:,:]
            img_l = out[:,1,:,:]
            X_l, X_h = pinglvyu(image)
            # print('x_h shape ', X_h.shape)
            high_loss = criterion[0](img_h, X_h)
            low_loss = criterion[0](img_l, X_l)
            loss_pinglv = high_loss + low_loss

            loss_vgg7, all_ap, all_an, loss_rec = 0, 0, 0, 0

            loss_rec = criterion[0](final_out, image)

            loss_vgg7, all_ap, all_an = criterion[1](final_out, image, adv_data)
            # print("rec_loss, vgg_loss, high_loss, low_loss ", loss_rec, loss_vgg7, high_loss, low_loss)
            real = vggloss(image)
            fake = vggloss(final_out)
            pertual_loss = criterion[0](fake, real)

            loss = loss_rec*10 + loss_vgg7 + loss_pinglv*2 + pertual_loss
            loss.backward()

            optim.step()
            losses.append(loss.cpu().item())
            rec_losses.append(loss_rec.cpu().item())
            vgg_losses.append(loss_vgg7.cpu().item())
            pinglv_losses.append(loss_pinglv.cpu().item())
            pertual_losses.append(pertual_loss.cpu().item())

        scheduler.step()

        print("epoch {}'s: loss:{:.5f}, rec_losses:{:.5f}, vgg_losses:{:.5f}, pinglv_losses:{:.5f}, pertual_losses:{:.5f}".format(epoch, np.sum(losses) / len(losses), \
                                                                                     np.sum(rec_losses) / len(rec_losses), \
                                                                                     np.sum(vgg_losses) / len(vgg_losses), np.sum(pinglv_losses) / len(pinglv_losses),\
                                                                                                                                 np.sum(pertual_losses) / len(pertual_losses)))

        if epoch % 1 == 0:
            torch.save(net.state_dict(), '{}/params_{}.pt'.format(model_dir, epoch))
            # torch.save(Dis.state_dict(), '{}/params_{}.pt'.format(Dis_dir, epoch))

        torch.save(net.state_dict(), '{}/params_pause.pt'.format(model_dir))
        print("Current epoch saved!")

        if epoch % 1 == 0:
            test(epoch, net, loader_test)


    torch.save(net.state_dict(), '{}/params_finished.pt'.format(model_dir))


def show_images(x, x_recon):
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1,i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1,2,0)))
        axes[0, i].set_title("Clean")

        axes[1, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)))
        axes[1, i].set_title("Recon")
    plt.axis("off")
    plt.savefig('./mnist_picture_result.png')
    print('picture already saved!')


def test(epoch, net, loader_test):
    net.eval()
    total = 0
    correct = 0

    for i, (images, labels) in enumerate(loader_test):
        inputs = images.cuda()
        y = labels.cuda()
        with torch.no_grad():
            pred, final_pred = net(inputs)
            logit = classifier(final_pred)
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

    model_dir = './mnist_lr_two'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    model_name = 'test2'
    epochs = 100
    # lr = 0.0001 #(0.0001, 0.001)
    lr = 0.001 #(0.0001, 0.001)
    loader_train = dataloader
    net = Dehaze(1, 2)
    # net.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/final_minst/Final_minst/params_45.pt'))
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
