import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
import dupty_model
# from primary_model import VAE
from primary_model_skip_attention import VAE
# from utils.dataset import *
from classifier import Classifier
from vaedataset import Dataset
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import torch.autograd as autograd
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

device = torch.device('cuda')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


# initialize model weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Block') == -1:
        m.weight.data.normal_(0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1, 0.02)
        m.bias.data.fill_(0)


class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma**self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]



# init parameters
model_dir = './Result/Fashion_Mnist5'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
image_size = 28
batch_size = 512
log_dir = 'log'
lr = 0.001
epochs = 90
beta = 0.1

# prepare dataset
data = np.load('/remote-home/cs_igps_yangjin/data/f_xs_mnist.npy')  # image data in npy file
labels = np.load('/remote-home/cs_igps_yangjin/data/f_ys_mnist.npy')  # labels data in npy file
adv_data = np.load('/remote-home/cs_igps_yangjin/data/f_advs_mnist.npy')  # adversarial image data in npy file
dataset = Dataset(data, labels, adv_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)


# create modules needed
model1 = dupty_model.VAE()
model1.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/Fashion_Mnist/Result/dupty_branch_changeTransform_epoch300/params_finished.pt'))
model1 = model1.cuda()

model = VAE()
model.apply(weights_init)
model = model.cuda()
# model.load_state_dict(torch.load('disentangled_model/disentangle_mnist_5/params_finished.pt'))
print('Using: ', torch.cuda.get_device_name(torch.cuda.current_device()))
# training module
model.train()
# model.train()

optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = MinExponentialLR(optimizer, gamma=0.998, minimum=1e-5)
MSECriterion = nn.MSELoss().to(device)

classifier = Classifier(28, 1)
classifier.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyVae3/Fashion_Mnist/classifier_f_mnist/params_finished.pt'))
classifier = classifier.cuda()

def classification_loss(recon, label, classifier):
    criterion = nn.CrossEntropyLoss()
    # get output
    output = classifier(recon)
    # calculate loss
    loss = criterion(output, label)

    return loss


def recon_loss_function(recon, target, distribution, step, beta=1):
    CE = MSECriterion(recon,target)
    normal = Normal(
        torch.zeros(distribution.mean.size()).to(device),
        torch.ones(distribution.stddev.size()).to(device))
    KLD = kl_divergence(distribution, normal).mean()
    return CE + beta * KLD, CE, KLD


def test(epoch):
    correct = 0.
    cost = 0.
    total = 0.
    e = 0
    for batch_idx, (images, labels, adv_data) in enumerate(dataloader):
        x = adv_data.cuda()
        y = labels.cuda()

        output, _, _, _, _, _, _ = model(x)
        logit = classifier(output)
        prediction = torch.max(logit, 1)[1]
        correct = correct + torch.eq(prediction, y).float().sum().item()

        loss = F.cross_entropy(logit, y, reduction='sum').item()
        cost = cost + loss

        total = total + x.size(0)

    accuracy = correct / total

    cost /= total
    print()
    print('TEST *TOP* ACC:{:.4f} at e:{:03d}'.format(accuracy, epoch))
    print()


# training steps
step = 0
for epoch in range(1, epochs + 1):
    print('Epoch: {}'.format(epoch))

    # init output lists
    vae_losses = list()
    recon_losses = list()
    recon_kls = list()
    classifier_losses = list()
    encoder_losses = list()

    datas = list()
    outputs = list()

    # loop for each data pairs
    for data, label, adv_data in dataloader:
        # initialize
        step += 1
        data = data.cuda()
        label = label.cuda()
        adv_data = adv_data.cuda()

        optimizer.zero_grad()
        '''
        additional
        '''



        # get data and run model
        output, mean, var, z, _, dsm, dss = model(adv_data)
        distribution = Normal(dsm, dss)

        recon_loss, img_recon, recon_kl = recon_loss_function(output, data, distribution, step, 0.1)

        # calculate losses
        c_loss = classification_loss(output, label, classifier)
        en_loss = torch.nn.functional.mse_loss(mean, model1.encode(data)[0]) + torch.nn.functional.mse_loss(var, model1.encode(data)[1])
        loss = recon_loss * 4 + c_loss * 10 + en_loss * 15
        # print(recon_loss, c_loss, en_loss)

        loss.backward()

        # clip for gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

        # step optimizer
        optimizer.step()

        # record results
        vae_losses.append(loss.cpu().item())
        recon_losses.append(recon_loss.cpu().item())
        classifier_losses.append(c_loss.cpu().item())
        recon_kls.append(recon_kl.cpu().item())
        encoder_losses.append(en_loss.cpu().item())
        datas.append(data.cpu())
        outputs.append(output.cpu())


    # print out loss
    print("batch {}'s: vae_loss:{:.5f}, recon_loss: {:.5f}, recon_kl: {:.5f}, classifier_loss: {:.5f}, encoder_loss: {:.5f}".format(step, np.sum(vae_losses)
                / len(vae_losses), np.sum(recon_losses) / len(recon_losses), np.sum(recon_kls) / len(recon_kls), np.sum(classifier_losses) / len(classifier_losses), np.sum(encoder_losses) / len(encoder_losses)))

    # step scheduler
    scheduler.step()
    # save model parameters
    if epoch % 5 == 0:
        torch.save(model.state_dict(), '{}/params_{}.pt'.format(model_dir, epoch))

    torch.save(model.state_dict(), '{}/params_pause.pt'.format(model_dir))
    print("Current epoch saved!")

    if epoch % 3 == 0:
        test(epoch)

torch.save(model.state_dict(), '{}/params_finished.pt'.format(model_dir))

