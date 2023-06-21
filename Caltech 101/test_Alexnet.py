import sys, os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.insert(0, os.path.abspath('..'))
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from test.attacks import *
from tqdm import tqdm
from utils.classifier import *
from new import test_dataset
from torchvision.models import alexnet


net = alexnet()
net.classifier[6] = nn.Linear(4096, 101)
net.load_state_dict(torch.load('./caltec_model/params_29.pt'))
net.eval()
net = net.cuda()


test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4, drop_last=True)

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