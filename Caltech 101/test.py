import torch
from torchvision import transforms
from Caltech101Data import Caltech101Data
from Cal_classifier import Classifier
from Trainer import Trainer
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler
from torch.utils.data import random_split

tr = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.Resize((224,224))])
model = Classifier(102) # or you can torch.load(model_complete.mdl)
cd = Caltech101Data('./101_ObjectCategories', tr)
print(len(cd))
optimizer = Adam(model.parameters())
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
loss_function = torch.nn.CrossEntropyLoss()
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

'''
# if you want to load a checkpoint of a model
checkpoint = torch.load('model_checkpoint.mdl')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
'''

dataset_size = {'train': 7000, 'val': 2144}
data = {}
data['train'], data['val'] = random_split(cd, [dataset_size['train'], dataset_size['val']])
loader = {phase: DataLoader(data[phase], batch_size=20) for phase in ['train', 'val']}

trainer = Trainer(loader, optimizer, loss_function, scheduler, model, device)

def main():
	trainer.train_with_validation(10, dataset_size)

if __name__ == '__main__':
	main()
