import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys
sys.path.append(os.pardir)
import torch,json
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from torchvision.models import alexnet
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from purifier_vae.adversarial_one import add_adv
from new import test_dataset
from UNet_model import Dehaze
# from purifier_network import Dehaze

device = torch.device('cuda')

# torch.set_num_threads(3)


classifier = alexnet()
classifier.classifier[6] = nn.Linear(4096, 101)
classifier.load_state_dict(torch.load('./caltec_model/params_29.pt'))
classifier.eval()
classifier = classifier.cuda()




test_dataloader = DataLoader(dataset=test_dataset, batch_size=6, shuffle=True, num_workers=4, drop_last=True)


model = Dehaze()
# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/Caltech101/Cal_Result/params_20.pt')) #不错结果
model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/Caltech101/Cal_Result/params_finished.pt')) #不错结果
model.eval()
model = model.cuda()


# def test_sample_acc():
#     step = 0
#     correct = 0
#     total = 0
#     for batch_idx, (data, label) in enumerate(test_dataloader):
#         step += 1
#         data = data.cuda()
#         label = label.cuda()
#
#         _, final_out = model(data)
#         logit = classifier(final_out)
#         prediction = torch.max(logit, 1)[1]
#
#         correct = correct + torch.eq(prediction, label).float().sum().item()
#         total = total + data.size(0)
#
#
#     accuracy = correct / total
#     print(total)
#     print('Classifier_ACC: ', accuracy)

def show_images(x, x_recon):
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1,i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1,2,0)))
        axes[0, i].set_title("Clean")

        axes[1, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)))
        axes[1, i].set_title("Recon")
    plt.axis("off")
    plt.savefig('./Caltech101_visi.png')
    print('picture already saved!')


def imgshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.cpu().numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)).astype(np.uint8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('./clean1.png')
    print('image already saved!')
    return


def imgshow1(img):
    img = img / 2 + 0.5     # unnormalize
    # img = img / 255
    npimg = img.cpu().numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)).astype(np.uint8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('./adv1.png')
    print('image already saved!')
    return

def imgshow2(img):
    img = img / 2 + 0.5     # unnormalize
    # img = img / 255
    npimg = img.cpu().numpy()
    # plt.imshow(np.transpose(npimg, (1, 2, 0)).astype(np.uint8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('./recon1.png')
    print('image already saved!')
    return


def test_advsample_acc():
    # adv_accuracy = {'pgd_n': 0, 'pgd_t': 0}
    # adv_accuracy = {'pgd_t': 0}
    # adv_accuracy = {'ddn_n': 0, 'jsma_t': 0, 'cw_n': 0}
    # adv_accuracy = {'jsma_t': 0}
    # adv_accuracy = {'cw_n': 0}
    # adv_accuracy = {'aa_n': 0}
    # adv_accuracy = {'fgsm': 0}
    # adv_accuracy = {'pgd_n': 0}
    adv_accuracy = {'bim': 0}
    # adv_accuracy = {'deepfool': 0}
    # adv_accuracy = {'l-bfgs': 0}

    for adv in adv_accuracy:
        true = 0
        total = len(test_dataloader)
        print('total ', total)
        correct = 0
        num = 0
        for image, label in test_dataloader:
            image = image.cuda()
            label = label.cuda()

            # get model output
            output, adv_out = add_adv(classifier, image, label, adv)
            output_class = classifier(output)

            _, final_out = model(adv_out)

            # out = torchvision.utils.make_grid(image)
            # out1 = torchvision.utils.make_grid(adv_out)
            # out2 = torchvision.utils.make_grid(final_out)
            # imgshow(out)
            # imgshow1(out1)
            # imgshow2(out2)
            # final_out = model(adv_out)
            adv_out_class = classifier(final_out)

            # show_images(image.data, final_out.data)
            prediction = torch.max(adv_out_class, 1)[1]
            correct = correct + torch.eq(prediction, label).float().sum().item()
            num = num + adv_out.size(0)
        accuracy = correct / num
        print(total)
        print('Classifier_ACC: ', accuracy)


            # get model predicted class
    #         true_class = torch.argmax(output_class, 1)
    #         adversarial_class = torch.argmax(adv_out_class, 1)
    #
    #         # print(f'attack method {adv}')
    #         # print(f'actual class {true_class}')
    #         # print(f'adversarial class {adversarial_class}')
    #
    #         # calculate number of correct classification
    #         true += torch.sum(torch.eq(true_class, adversarial_class))
    #
    #         print(int(true) / total)
    #     adv_accuracy[adv] = int(true) / total
    #     print('total: ', total)
    #     print('int(true): ', int(true))
    #     print(int(true) / total)
    #     print('=================================')
    # print()

    # with open(f'./accuracy.txt', 'w') as f:
    #     json.dump(adv_accuracy, f)

if __name__ == '__main__':
    # test_sample_acc()
    # eval()
    test_advsample_acc()