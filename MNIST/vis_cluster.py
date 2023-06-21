import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
sys.path.append(os.pardir)
import torch,json
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from classifier import Classifier
# from black_box_model import model_e
import umap
from torchvision import datasets
from torchvision import transforms
from purifier_vae.adversarial_one import add_adv
from mnist_model_two import Dehaze #最好


device = torch.device('cuda')



classifier = Classifier(28, 1)
classifier.load_state_dict(torch.load('classifier_mnist.pt'))
classifier = classifier.cuda()



transform = transforms.Compose([transforms.ToTensor()])
test_data = torchvision.datasets.MNIST(root='./MNIST_Test_Data', train=False, transform=transform, download=False)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=1, shuffle=True)


model = Dehaze(1, 2)
model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/final_minst/Final_minst_lr/params_2.pt')) #不错结果 最好
model = model.cuda()

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
            # output, adv_out = add_adv(classifier, image, label, adv, default=False)
            #
            _, final_out = model(image)
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
        adv_examples = np.load('/remote-home/cs_igps_yangjin/adversarial dataset/advs_mnist_v2.npy')
        adv_examples = adv_examples.reshape(adv_examples.shape[0], 28 * 28)
        sample = np.random.randint(adv_examples.shape[0], size=3000)
        adv_examples = adv_examples[sample, :]

        adv_images = torch.from_numpy(adv_examples.reshape(adv_examples.shape[0], 1, 28, 28))
        adv_images = adv_images.cuda()
        _, def_out = model(adv_images)
        labels = classifier(def_out)
        labels = torch.argmax(labels, 1).detach().cpu().numpy()

        def_out = def_out.detach().cpu().numpy()
        def_out = def_out.reshape(def_out.shape[0], 28 * 28)

        fit = umap.UMAP(n_components=2, random_state=42)
        u = fit.fit_transform(def_out.reshape(def_out.shape[0], 28 * 28))

        plt.scatter(u[:, 0], u[:, 1], c=labels, cmap='Spectral', s=14)
        plt.gca().set_aspect('equal', 'datalim')
        clb = plt.colorbar(boundaries=np.arange(11) - 0.5)
        clb.set_ticks(np.arange(10))
        clb.ax.tick_params(labelsize=18)
        plt.xticks([])
        plt.yticks([])
        # plt.title(f'MNIST clustering under {adv.upper()}', fontsize=24);
        plt.title(f'MNIST(With Defense)', fontsize=16);
        if not os.path.exists('./img'):
            os.makedirs('./img')
        plt.savefig(f'img/{adv}_pure3.png', dpi=300, pad_inches=0)
        plt.clf()


if __name__ == '__main__':
    # test_sample_acc()
    # eval()
    test_advsample_acc()