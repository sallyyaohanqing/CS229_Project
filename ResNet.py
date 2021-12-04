
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import sklearn
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def accuracy(output, labels):
    _,pred = torch.max(output, dim=1)
    return torch.sum(pred==labels).item()

#num_folder: the number of folders of images to use as the whole dataset
#num_epochs: the number of epochs to run the ResNet
#batch_size: the batch size to be used for Gradient Descent
#report_freq: frequency for reporting training status
def main(num_folder,num_epochs,batch_size,report_freq):
	X_cnn_data,Y_cnn_label=[],[]
	# Create a dictionary where key value is the emotion and value associated
	label_dict={"AF":0,"AN":1,"DI":2,"HA":3,"NE":4,"SA":5,"SU":6}

	# Construct normalization transformation so that the image has
	# mean [0.485, 0.456, 0.406] and SD [0.229, 0.224, 0.225]
	transform = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	# Load in data by looping
	image_dir="/Users/hanqingyao/Downloads/KDEF_masked_all"
	image_subdirs=[x[0] for x in os.walk(image_dir)][1:]
	for subdir in image_subdirs[:num_folder]:
		files = os.walk(subdir).__next__()[2]
		for file in files:
			if (file.find("surgical_blue")!=-1)|(file.find("surgical_green")!=-1):
				continue
			im=cv2.imread(os.path.join(subdir,file))

			Y_cnn_label.append(label_dict[file[4:6]])
			
			im=cv2.resize(im,(64,64))/255

			X_cnn_data.append(transform(im))


	X_cnn_data = np.stack(X_cnn_data)
	Y_cnn_label = np.stack(Y_cnn_label)
	# 80% goes to training, 20% for validation
	X_cnn_train,X_cnn_test,Y_cnn_train,Y_cnn_test=train_test_split(X_cnn_data,Y_cnn_label,test_size=0.2)


	X_train_dataloader = DataLoader(X_cnn_train, batch_size, shuffle=False)
	Y_train_dataloader = DataLoader(Y_cnn_train, batch_size, shuffle=False)
	X_test_dataloader = DataLoader(X_cnn_test, batch_size, shuffle=False)
	Y_test_dataloader = DataLoader(Y_cnn_test, batch_size, shuffle=False)

	# Check for available resources
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Construct the ResNet18 model
	resnet = torchvision.models.resnet18(pretrained=True)
	resnet = resnet.float() # cast


	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(resnet.parameters(), lr=0.0001, momentum=0.9)

	# Add the output linear layer
	num_ftrs = resnet.fc.in_features
	resnet.fc = nn.Linear(num_ftrs, 7) # we have 7 class labels


	valid_loss_min = np.Inf # Keep track of the current minimum loss reached through all epochs
	train_loss, val_loss = [], []
	train_acc, val_acc = [], []
	total_step = len(X_train_dataloader)
	# Loop through each epoch
	for epoch in range(1, num_epochs+1):
		training_loss = 0.0
		correct_train = 0
		total_train=0
		print(f'Epoch {epoch}\n')
		num_iter=0
		for (data_train, label_train) in zip(X_train_dataloader, Y_train_dataloader):
			# send the input and label to the available device
			data_train, label_train = data_train.to(device), label_train.to(device)
			optimizer.zero_grad() # zero out the parameter gradients

			outputs = resnet(data_train.float()) # cast to same data type
			loss_train = criterion(outputs, label_train)
			# Perform backward propagation and optimize step during the training process
			loss_train.backward()
			optimizer.step()

			training_loss += loss_train.item() # Accumulate the training loss
			correct_train += accuracy(outputs,label_train) # Accumulate the number of correctly predicted labels
			total_train += batch_size
			if (num_iter) % report_freq == 0:
				print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
					.format(epoch, num_epochs, num_iter, total_step, loss_train.item()))
			num_iter+=1
		# Report the training accuracy and loss
		train_acc.append(100 * correct_train / total_train)
		train_loss.append(training_loss/total_step)
		print(f'\nTraining Loss: {np.mean(train_loss):.4f}, Training Accuracy: {(100 * correct_train/total_train):.4f}')
		
		validation_loss = 0
		total_valid=0
		correct_valid=0
		# Suppress learning on the validation data
		with torch.no_grad():
			resnet.eval()
			for data_valid, label_valid in zip(X_test_dataloader,Y_test_dataloader):
				data_valid, label_valid = data_valid.to(device), label_valid.to(device)
				outputs_valid = resnet(data_valid.float())
				loss_valid = criterion(outputs_valid, label_valid)
				validation_loss += loss_valid.item()
				correct_valid += accuracy(outputs_valid,label_valid)
				total_valid += batch_size
			# Report the validation accuracy and loss
			val_acc.append(100 * correct_valid/total_valid)
			val_loss.append(validation_loss/len(Y_test_dataloader))
			network_learned = validation_loss < valid_loss_min
			print(f'Validation Loss: {np.mean(val_loss):.4f}, Validation Accuracy: {(100 * correct_valid/total_valid):.4f}\n')

	     	# Update the network structure with the minimum loss
			if network_learned:
				valid_loss_min = validation_loss
				torch.save(resnet.state_dict(), 'ResNet.pt')
				print('Getting an improved model. Updating the current best network structure.')
		resnet.train()


if __name__ == '__main__':
	main(num_folder=100,num_epochs=30,batch_size=60,report_freq=10)








