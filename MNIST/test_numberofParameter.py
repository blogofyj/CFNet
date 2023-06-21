import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from torchsummary import summary
from mnist_model import Dehaze

if __name__ == "__main__":
    model = Dehaze(1, 2).cuda()
    summary(model, input_size=(1,28,28))