import os, os.path, sys
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import logging
import torch
device = torch.device('cuda')
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn
import torchvision
from torchvision import transforms
from torchvision.models import alexnet
from torchvision.models import vgg16
from torchvision.models import resnet18, resnet50
from PIL import Image
from tqdm import tqdm
import copy
import time
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import VisionDataset, ImageFolder
from caltech_dataset import Caltech



NETWORK_TYPE = 'alexnet'
BATCH_SIZE = 256
if NETWORK_TYPE == 'vgg' or NETWORK_TYPE == 'resnet':
    BATCH_SIZE = 16

LR = 1e-3
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-5

NUM_EPOCHS = 30
STEP_SIZE = 20
GAMMA = 0.1

LOG_FREQUENCY = 10
PRETRAINED = True
FREEZE = 'conv_layers'
# BEST_NET = False
RANDOM = 42
TRAIN_SIZE = 0.5

if PRETRAINED:
    mean, stdev = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
else:
    mean, stdev = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

train_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, stdev)
                                    ])

# Define transforms for the evaluation phase
eval_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean, stdev)
                                    ])

DATA_DIR = '../Caltech101/'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None, split='train', transform='None') :
    # dir = 'Caltech101'
    images = []
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return x.lower().endswith(extensions)

    inputFile = os.path.join(dir, split + '.txt') # 'Caltech101/{split}.txt'
    with open(inputFile, 'r') as f:
      input_images = f.read().splitlines()

    root = os.path.join(dir, '101_ObjectCategories/') # 'Caltech101/101_ObjectCategories/'

    for fname in input_images:
      fpath = os.path.split(fname)
      # print(fpath) # 'accordion' 'image_0002.jpg'
      target = fpath[0] # 'accordion'
      path = os.path.join(root, fname) # 'Caltech101/101_ObjectCategories/accordion/image_0002.jpg'
      if is_valid_file(path) and target != 'BACKGROUND_Google':
        item = (path, class_to_idx[target])
        images.append(item)

    return images


class Caltech(VisionDataset):
    ''' Caltech 101 Dataset '''
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = split
        classes, class_to_idx = self._find_classes(self.root)
        samples = make_dataset(self.root, class_to_idx, IMG_EXTENSIONS, split=self.split, transform=transform)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n""Supported extensions are: " + ",".join('extensions')))

        self.loader = pil_loader
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        root = os.path.join(dir, '101_ObjectCategories/')
        classes = [d.name for d in os.scandir(root) if d.is_dir()]
        classes.remove('BACKGROUND_Google')
        classes.sort()
        # print(classes)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        # print(class_to_idx)
        return classes, class_to_idx

    def __getitem__(self, index):
        path, label = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            image = self.transform(sample)
        return image, label

    def __len__(self):
        return len(self.samples)

train_dataset = Caltech(DATA_DIR, split='train', transform=train_transform)
test_dataset = Caltech(DATA_DIR, split='test', transform=eval_transform)


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


def imgshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    plt.savefig('./caltech101.png')
    print('image already saved!')
    return

# x, y = next(iter(DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=4, drop_last=True)))
# out = torchvision.utils.make_grid(x)
# imgshow(out)


if NETWORK_TYPE == 'alexnet':
    net = alexnet(pretrained=True)
    net.classifier[6] = nn.Linear(4096, 101)

criterion = nn.CrossEntropyLoss()

if FREEZE == 'conv_layers':
    parameters_to_optimize = net.classifier.parameters()
else:
    raise (ValueError(f"Error Freezing layers (FREEZE = {FREEZE}) \n Possible values are: 'no_freezing', 'conv_layers', 'fc_layers' "))

optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
EVAL_ACCURACY_ON_TRAINING = True
criterion_val = nn.CrossEntropyLoss(reduction='sum')


def train(net):
    start = time.time()
    net = net.to(device)  # bring the network to GPU if DEVICE is cuda
    cudnn.benchmark  # Calling this optimizes runtime

    # save best config
    best_net = 0
    best_epoch = 0
    best_val_acc = 0.0

    # save accuracy and loss
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []

    current_step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}, LR = {scheduler.get_lr()}")

        net.train()  # Sets module in training mode

        running_corrects_train = 0
        running_loss_train = 0.0

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                outputs_train = net(images)

                _, preds = torch.max(outputs_train, 1)

                loss = criterion(outputs_train, labels)

                # Log loss
                if current_step % LOG_FREQUENCY == 0:
                    print('Step {}, Loss {}'.format(current_step, loss.item()))

                # Compute gradients for each layer and update weights
                loss.backward()  # backward pass: computes gradients
                optimizer.step()  # update weights based on accumulated gradients

            current_step += 1

        # store loss and accuracy values
        running_corrects_train += torch.sum(preds == labels.data).data.item()
        running_loss_train += loss.item() * images.size(0)

        train_acc = running_corrects_train / float(len(train_dataset))
        train_loss = running_loss_train / float(len(train_dataset))

        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        print(f'train_accuracies {np.sum(train_accuracies)/len(train_accuracies)}, train_losses is {np.sum(train_losses)/len(train_losses)}')

        torch.save(net.state_dict(), '{}/params_{}.pt'.format(model_dir, epoch))
        if epoch % 3 == 0:
            test(net)

        scheduler.step()



def test(net):
    net = net.to(device)  # this will bring the network to GPU if DEVICE is cuda
    net.train(False)  # Set Network to evaluation mode

    running_corrects = 0
    for images, labels in tqdm(test_dataloader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward Pass
        outputs = net(images)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Update Corrects
        running_corrects += torch.sum(preds == labels.data).data.item()

    # Calculate Accuracy
    accuracy = running_corrects / float(len(test_dataset))

    print()
    print(f"Test Accuracy: {accuracy}")


if __name__ == '__main__':
    model_dir = './caltec_model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    train(net)

