import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch,json
import torch.nn as nn
import torch.utils.data as Data
import torchvision  # 数据库
import torch.nn.functional as F
from matplotlib import pyplot as plt
import numpy as np
from resnet_model import ResNet18
from torchvision import datasets
from torchvision import transforms
from adversarial_one import add_adv
# from Fuliye_model import Dehaze  #不错结果

# from cifar_final_model import Dehaze
from Fu_final_model import Dehaze
# from purifier_network import Dehaze

device = torch.device('cuda')

# torch.set_num_threads(3)


classifier = ResNet18()
classifier.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/DC-VAE-main/models/Result/ResNet18/params_finished.pt'))
classifier = classifier.cuda()



transform = transforms.Compose([transforms.ToTensor()])
test_data = torchvision.datasets.CIFAR10(root='./cifar10', train=False, transform=transform, download=False)
testloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=64, shuffle=True)


model = Dehaze(3, 6)
# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/purifier_vae/purifier_network_fuliye_test/params_finished.pt')) #不错结果
# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/purifier_vae/purifier_network_fuliye_test/params_75.pt')) #不错结果

# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/purifier_vae/purifier_network3/params_finished.pt'))
# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/purifier_vae/purifier_network_fuliye/params_finished.pt'))
# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/purifier_vae/purifier_network_fuliye_Two/params_finished.pt'))
# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/purifier_vae/purifier_network_fuliye_Final/params_finished.pt'))
# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/purifier_vae/cifar_final_model/params_28.pt'))
# model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/purifier_vae/Cifar10_No_VGGLossAndPinglv/params_100.pt')) #18 #28 model4 38 最好
model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/purifier_vae/Cifar10_final_model4/params_20.pt'))
model = model.cuda()



class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layer):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layer = extracted_layer

    def forward(self, x):
        # x = x.cuda()
        outputs = []
        for name, module in self.submodule._modules.items():
            # print(x.shape)
            print(name)
            print(module)
            if name in ["e_module", "img_module"]:
                continue
            x = module(x)
            if name in self.extracted_layer:
                outputs.append(x.cpu())
        return outputs



def Feature_visual(outputs):
    for i in range(len(outputs)):
        out = outputs[i].cpu().data.numpy()
        feature_img = out[:, 0, :, :].squeeze()  # 选择第一个特征图进行可视化
        feature_img = np.asarray(feature_img * 255, dtype=np.uint8)
        plt.imshow(feature_img, cmap='gray')
        plt.show()

        print('finish')


def eval():
    # model.eval()
    with torch.no_grad():
        # extract_feature = FeatureExtractor(model, extracted_layer)(test_x)
        features = model.getFeatures(adv_image)
        Feature_visual(features)
        # output, _, _, _, test_out = model(adv_image)  # 输入测试集
        #
        #
        # output = classifier(output)
        # # 获得当前softmax层最大概率对应的索引值
        # pred = torch.max(output, 1)[1]
        # # 将二维压缩为一维
        # # pred_y = pred.data.numpy().squeeze()
        # pred_y = pred.data.cpu().numpy().squeeze()
        # # label_y = test_y.data.numpy()
        # label_y = test_y.data.cpu().numpy()
        # print(pred_y)
        # print(label_y)
        # accuracy = sum(pred_y == label_y) / test_y.size()
        # print("准确率为 %.2f" % (accuracy))



def test_sample_acc():
    step = 0
    correct = 0
    total = 0
    for batch_idx, (data, label) in enumerate(testloader):
        step += 1
        data = data.cuda()
        label = label.cuda()

        _, final_out = model(data)
        logit = classifier(final_out)
        prediction = torch.max(logit, 1)[1]

        correct = correct + torch.eq(prediction, label).float().sum().item()
        total = total + data.size(0)


    accuracy = correct / total
    print(total)
    print('Classifier_ACC: ', accuracy)


def show_images(x, x_recon):
    fig, axes = plt.subplots(2, 5, figsize=(10, 6))
    for i in range(5):
        axes[0, i].axis("off"), axes[1,i].axis("off")
        axes[0, i].imshow(x[i].cpu().numpy().transpose((1,2,0)))
        axes[0, i].set_title("Clean")

        axes[1, i].imshow(x_recon[i].cpu().numpy().transpose((1,2,0)))
        axes[1, i].set_title("Recon")
    plt.axis("off")
    plt.savefig('./cifarfinal.png')
    print('picture already saved!')


def test_advsample_acc():
    # adv_accuracy = {'pgd_n': 0, 'pgd_t': 0}
    adv_accuracy = {'pgd_n': 0}
    # adv_accuracy = {'ddn_n': 0}
    # adv_accuracy = {'ddn_n': 0}
    # adv_accuracy = {'pgd_t': 0}
    # adv_accuracy = {'ddn_n': 0, 'jsma_t': 0, 'cw_n': 0}
    # adv_accuracy = {'jsma_t': 0}
    # adv_accuracy = {'fgsm': 0}
    # adv_accuracy = {'aa_n': 0}
    # adv_accuracy = {'cw_n': 0}

    for adv in adv_accuracy:
        true = 0
        total = len(test_data)
        print('total ', total)
        print(f'attack method {adv}')
        for image, label in testloader:
            image = image.cuda()
            label = label.cuda()

            # get model output
            output, adv_out = add_adv(classifier, image, label, adv, default=False)
            output_class = classifier(output)

            _, final_out = model(adv_out)
            # final_out = model(adv_out)
            adv_out_class = classifier(final_out)

            # show_images(image.data, final_out.data)

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

    with open(f'./accuracy.txt', 'w') as f:
        json.dump(adv_accuracy, f)

if __name__ == '__main__':
    test_sample_acc()
    # eval()
    # test_advsample_acc()