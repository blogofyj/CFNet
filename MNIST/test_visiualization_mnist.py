import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
sys.path.append(os.pardir)
import torch,json
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库
import torch.nn.functional as F
from matplotlib import pyplot as plt
# import numpy as np
#from classifier import Classifier

# from black_box_model import model_e
import time
# from torchvision import datasets
from torchvision import transforms
from purifier_vae.adversarial_one import add_adv
from mnist_model_two import Dehaze #最好


# from model_three import Dehaze
# from purifier_network import Dehaze

device = torch.device('cuda')

# torch.set_num_threads(3)
class Basicblock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(Basicblock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes),
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.block1 = self._make_layer(block, 16, num_block[0], stride=1)
        self.block2 = self._make_layer(block, 32, num_block[1], stride=2)
        self.block3 = self._make_layer(block, 64, num_block[2], stride=2)
        # self.block4 = self._make_layer(block, 512, num_block[3], stride=2)

        self.outlayer = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_block, stride):
        layers = []
        for i in range(num_block):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.block1(x)                       # [200, 64, 28, 28]
        x = self.block2(x)                       # [200, 128, 14, 14]
        x = self.block3(x)                       # [200, 256, 7, 7]
        # out = self.block4(out)
        x = F.avg_pool2d(x, 7)                   # [200, 256, 1, 1]
        x = x.view(x.size(0), -1)                # [200,256]
        out = self.outlayer(x)
        return out

ResNet18 = ResNet(Basicblock, [1, 1, 1, 1], 10)

# classifier = Classifier(28, 1)
# classifier.load_state_dict(torch.load('classifier_mnist.pt'))
# classifier = classifier.cuda()

# classifier = model_e()
classifier = ResNet18
# classifier.load_state_dict(torch.load('./param_pretrained/Model_E_mnist_params.pt'))
classifier.load_state_dict(torch.load('./classifier_res18/params_98.pt'))
classifier = classifier.cuda()



transform = transforms.Compose([transforms.ToTensor()])
test_data = torchvision.datasets.MNIST(root='./MNIST_Test_Data', train=False, transform=transform, download=False)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=True)


model = Dehaze(1, 2)
# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/final_minst/Final_minst_Two/params_5.pt'))
# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/final_minst/Final_minst_Three_rec10_pl2/params_pause.pt'))
model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/final_minst/Final_minst_lr/params_2.pt')) #不错结果 最好
# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/final_minst/Final_minst_lr/params_4.pt')) #不错结果

# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/final_minst/mnist_lr_two/params_3.pt'))
model = model.cuda()



# def show_images(x, x_recon):
#     fig, axes = plt.subplots(2, 5, figsize=(10, 6))
#     for i in range(5):
#         axes[0, i].axis("off"), axes[1,i].axis("off")
#         axes[0, i].imshow(x[i].cpu().numpy().transpose((1,2,0)))
#         axes[0, i].set_title("Clean")
#
#         axes[1, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)))
#         axes[1, i].set_title("Recon")
#     plt.axis("off")
#     plt.savefig('./mnist_fgsm_1.png')
#     print('picture already saved!')

def show_images(x, x_adv,x_recon):
    fig, axes = plt.subplots(3, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1,i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1,2,0)))
        axes[0, i].set_title("Clean")

        axes[1, i].imshow(x_adv[i].cpu().numpy().transpose((1,2,0)))
        axes[1, i].set_title("Adv")

        axes[2, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)))
        axes[2, i].set_title("Recon")
    plt.axis("off")
    plt.savefig('./new1.png')
    print('picture already saved!')


def test_sample_acc():
    # adv_accuracy = {'pgd_n': 0}
    # adv_accuracy = {'i-fgsm': 0}
    adv_accuracy = {'fgsm': 0}

    for adv in adv_accuracy:
        true = 0
        total = len(test_data)
        print('total ', total)
        correct = 0
        num = 0
        print('adv', adv)
        for image, label in testloader:
            image = image.cuda()
            label = label.cuda()

            # get model output
            output, adv_out = add_adv(classifier, image, label, adv, default=False)
            #
            _, final_out = model(adv_out)
            adv_out_class = classifier(final_out)

            prediction = torch.max(adv_out_class, 1)[1]
            correct = correct + torch.eq(prediction, label).float().sum().item()
            num = num + image.size(0)

        accuracy = correct / total
        print(total)
        print('Classifier_ACC: ', accuracy)

def test_advsample_acc():
    # adv_accuracy = {'pgd_n': 0, 'pgd_t': 0}
    # adv_accuracy = {'pgd_n': 0}
    # adv_accuracy = {'ddn_n': 0}
    # adv_accuracy = {'cw_n': 0}
    # adv_accuracy = {'pgd_t': 0}
    # adv_accuracy = {'ddn_n': 0, 'jsma_t': 0, 'cw_n': 0}
    # adv_accuracy = {'jsma_t': 0}
    # adv_accuracy = {'aa_n': 0}
    adv_accuracy = {'fgsm': 0}
    # adv_accuracy = {'i-fgsm': 0}

    for adv in adv_accuracy:
        true = 0
        total = len(test_data)
        print('total ', total)
        correct = 0
        num = 0
        print('adv', adv)
        for image, label in testloader:
            image = image.cuda()
            label = label.cuda()

            # get model output
            output, adv_out = add_adv(classifier, image, label, adv, default=False)
            output_class = classifier(output)

            start_time = time.time()
            _, final_out = model(adv_out)
            end_time = time.time()
            print('time', end_time-start_time)
            break
            # final_out = model(adv_out)
            adv_out_class = classifier(final_out)

            # show_images(image.data, adv_out.data, final_out.data)
        #     prediction = torch.max(adv_out_class, 1)[1]
        #     correct = correct + torch.eq(prediction, label).float().sum().item()
        #     num = num + adv_out.size(0)
        # accuracy = correct / num
        # print(total)
        # print('Classifier_ACC: ', accuracy)


            # get model predicted class
            true_class = torch.argmax(output_class, 1)
            adversarial_class = torch.argmax(adv_out_class, 1)

            # print(f'attack method {adv}')
            # print(f'actual class {true_class}')
            # print(f'adversarial class {adversarial_class}')

            # calculate number of correct classification
            true += torch.sum(torch.eq(true_class, adversarial_class))

            print(int(true) / total)
        adv_accuracy[adv] = int(true) / total
        print('total: ', total)
        print('int(true): ', int(true))
        print(int(true) / total)
        print('=================================')
    print()

    # with open(f'./accuracy.txt', 'w') as f:
    #     json.dump(adv_accuracy, f)

if __name__ == '__main__':
    test_sample_acc()
    # eval()
    # test_advsample_acc()