import torch
from torch.optim import Adam, lr_scheduler
from Cal_classifier import Classifier
from torch.utils.data import DataLoader
import time
import copy

class Trainer():
	def __init__(self, loader, optimizer, loss_function, scheduler, model, device):
		self.model = model
		self.loader = loader
		self.optimizer = optimizer
		self.loss_function = loss_function
		self.device = device
		self.model = model.to(self.device)
		self.scheduler = scheduler

		if device == torch.device('cuda:0'):
			print('model in gpu')
			self.model.cuda()

	def train_with_validation(self, epochs, dataset_size):
		'''
		adapted from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
		'''
		since = time.time()

		model_wts = copy.deepcopy(self.model.state_dict())
		best_acc = 0.00
		
		for epoch in range(epochs):
			print('Epoch : {}/{}'.format(epoch, epochs-1))
			print('-'*10)

			for phase in ["train","eval"]:
				if phase=="train":
					self.model.train()
					self.scheduler.step()
				else:
					self.model.eval()
		
				running_loss = 0.0
				running_corrects = 0

				for batch_idx, data in enumerate(self.loader[phase],0):
					images, labels = data['image'], data['label']
					images = images.to(self.device)
					labels = labels.to(self.device)
					
					with torch.set_grad_enabled(phase == 'train'):
						outputs = self.model(images)
						_, preds = torch.max(outputs, 1)
						loss = self.loss_function(outputs, labels)

						if phase == "train":
							loss.backward()
							self.optimizer.step()
					running_loss += loss.item() * images.size(0)
					running_corrects += torch.sum(preds == labels.data)

					torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'loss': loss}, 'model_checkpoint.mdl')

			epoch_loss = running_loss/dataset_size[phase]
			epoch_corrects = running_corrects.double()/dataset_size[phase]

			print("{} epoch_loss: {:.3f} epoch_acc: {:.3f}".format(phase, epoch_loss, epoch_corrects))

			if phase == "eval" and epoch_corrects > best_acc:
				best_acc = epoch_corrects
				best_model_wts = copy.deepcopy(self.model.state_dict())

			print()

		time_elapsed = time.time() - since
		print("training completed in: {:.f}m {:.f}s".format(time_elapsed//60, time%60))

		print("Best accuracy: {:.4f}".format(best_acc))

		self.model.load_state_dict(best_model_wts)
		return self.model

	def train(self,epochs, loss=None):
		for epoch in range(epochs):
			running_loss = 0.0
			for batch_idx, data in enumerate(self.loader,0):
				print(self.device)
				inputs, labels = data['image'], data['label']
				inputs = inputs.to(self.device)
				labels = labels.to(self.device)
				self.optimizer.zero_grad()

				print("labels: ", type(labels))
				outputs = self.model(inputs)
				loss = self.loss_function(outputs, labels)
				loss.backward()
				self.optimizer.step()

				running_loss += loss.item()

				if batch_idx%2000==0:
					print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss/2000))
					running_loss = 0.0

				# save the model state in each beatch so that it is possible to resume training later
				torch.save({'epoch': epoch, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'loss': loss}, 'model_checkpoint.mdl')
		print('finished training')
		# we can save the whole model
		torch.save(self.model, 'model_complete.mdl')


