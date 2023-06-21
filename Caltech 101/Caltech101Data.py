import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import joblib

class Caltech101Data(Dataset):
	""" Caltech101 dataset """

	def __init__(self, inpath, transform=None):
		self.transform = transform
		self.path = inpath
		self.image_list = os.listdir(self.path)

	def __len__(self):
		return len(self.image_list)

	def __getitem__(self, idx):
		print(self.image_list[idx])
		image_path = os.path.join(self.path,self.image_list[idx])
		image = Image.open(image_path)
		image = image.convert('RGB')
		# label = self.image_list[idx].split('_image')[0]
		label = self.image_list[idx]
		print('label', label)
		int_label = joblib.load('labels')
		label = int_label.index(label)
		
		extra_transforms= transforms.Compose([transforms.ToTensor()])
		if self.transform:
			image = self.transform(image)
		image = extra_transforms(image)
		sample = {'image': image, 'label': label}
		
		return sample

if __name__ == '__main__':
	Caltech101Data('./101_ObjectCategories')