import numpy as np
import cv2
import torch
from purifier_network import Dehaze
import os
from train_purifier import Dataset
import torchvision
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda')

model = Dehaze(3, 3)
model.load_state_dict(torch.load('/remote-home/cs_igps_yangjin/MyMAD_VAE/Project/purifier_vae/purifier_network2/params_finished.pt'))
model = model.cuda()


data = np.load('/remote-home/cs_igps_yangjin/cifar10_data/xs_cifar10.npy')  # image data in npy file
labels = np.load('/remote-home/cs_igps_yangjin/cifar10_data/ys_cifar10.npy')  # labels data in npy file
adv_data = np.load('/remote-home/cs_igps_yangjin/cifar10_data/advs_cifar10.npy')  # adversarial image data in npy file
dataset = Dataset(data, labels, adv_data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
cifar_data = torch.from_numpy(adv_data)[0]


def visual(cifar_data, dir):
    image = cifar_data
    image = image.reshape(-1, 1024)
    r = image[0,:].reshape(32, 32)
    g = image[1,:].reshape(32, 32)
    b = image[2,:].reshape(32, 32)
    img = np.zeros((32,32,3))
    #将rgb还原彩色图像
    img[:,:,0] = r
    img[:,:,1] = g
    img[:,:,2] = b
    cv2.imwrite(dir, img)
    print("Done!")

def denorm(img, mean, std):
    img = img.clone().detach()
    # img shape is B, 3,64,64 and detached
    for i in range(3):
        img[:, i,:,:] *= std[i]
        img[:, i,:,:] += mean[i]
    return img

def disp_images(img, fname, nrow, norm="none"):
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))
    bs = img.shape[0]
    imsize = img.shape[2]
    nc = img.shape[1]
    if nc==3 and norm=="0.5":
        img = denorm(img,mean=[0.5, 0.5, 0.5], std=[0.5,0.5,0.5])
    elif nc==3 and norm=="none":
        pass
    elif nc==1:
        img = img
    else:
        raise ValueError("image has incorrect channels")
    img = img.view(bs,-1,imsize,imsize).cpu()
    grid = torchvision.utils.make_grid(img,nrow=nrow)
    torchvision.utils.save_image(grid, fname)


for image, label, adv_data in dataloader:
    image = image.cuda()
    adv_data = adv_data.cuda()
    out = model(adv_data)
    # out = out.cpu().data.numpy()
    disp_images(adv_data, './original.png', image.size(0), norm="0.5")
    disp_images(out, './denoising.png', image.size(0), norm="0.5")
    break


# if __name__ == "__main__":
#     original_dir = "./original.jpg"
#     visual(cifar_data, original_dir)
#     denoising_dir = "./denoising.jpg"
#     visual(out, denoising_dir)