import torch
from torch import nn
from torch.nn import functional

class ConvUnit(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(ConvUnit, self).__init__()
		self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=(1,1))
		self.norm = nn.BatchNorm2d(num_features=out_channels)
		self.acti = nn.ReLU()
	
	def forward(self, inputs):
		output = self.conv(inputs)
		output = self.norm(output)
		output = self.acti(output)

		return output

class Classifier(nn.Module):
	def __init__(self, num_classes):
		super(Classifier, self).__init__()
		self.num_classes = num_classes
		self.unit0 = ConvUnit(3, 32) # input shape (3,224,224)
		self.unit1 = ConvUnit(32, 32) # (32, 224, 224)
		self.pool0 = nn.MaxPool2d(2, stride=1) # (32, 112, 112)
		self.unit2 = ConvUnit(32, 64) #(64, 112, 112)
		self.unit3 = ConvUnit(64, 32) #(32, 112, 112)
		self.pool1 = nn.MaxPool2d(2, stride=1) #(32, 56, 56)
		self.unit4 = ConvUnit(32, 16) #(32, 56, 56)
		self.pool2 = nn.MaxPool2d(2, stride=1) #(16, 28, 28)
		self.network = nn.Sequential(self.unit0, self.unit1, self.pool0, self.unit2, self.unit3, self.pool1, self.unit4, self.pool2)

		# print(self.network)
		self.fc = nn.Linear(in_features=(781456), out_features=num_classes)

	def forward(self, input_images):
		output = self.network(input_images)
		output = output.reshape(output.size(0), -1)
		output = self.fc(output)

		return output
