
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
for subdir in image_subdirs[:10]:
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


X_train_dataloader = DataLoader(X_cnn_train, batch_size=60, shuffle=False)
Y_train_dataloader = DataLoader(Y_cnn_train, batch_size=60, shuffle=False)
X_test_dataloader = DataLoader(X_cnn_test, batch_size=60, shuffle=False)
Y_test_dataloader = DataLoader(Y_cnn_test, batch_size=60, shuffle=False)

# Check for available resources
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Construct the ResNet18 model
resnet = torchvision.models.resnet18(pretrained=True)
resnet=resnet.float() # cast


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet.parameters(), lr=0.0001, momentum=0.9)

# Add the output linear layer
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 7) # we have 7 class labels


n_epochs = 30
print_every = 10
valid_loss_min = np.Inf
val_loss = []
val_acc = []
train_loss = []
train_acc = []
total_step = len(X_train_dataloader)
# Loop through each epoch
for epoch in range(1, n_epochs+1):
	running_loss = 0.0
	correct = 0
	total=0
	print(f'Epoch {epoch}\n')
	num_iter=0
	for (data_train, target_train) in zip(X_train_dataloader, Y_train_dataloader):
		data_train, target_train = data_train.to(device), target_train.to(device)
		optimizer.zero_grad()

		outputs = resnet(data_train.float()) # cast to same data type
		loss = criterion(outputs, target_train)
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
		correct += accuracy(outputs,target_train)
		total += target_train.size(0)
		if (num_iter) % 5 == 0:
			print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
				.format(epoch, n_epochs, num_iter, total_step, loss.item()))
		num_iter+=1
	# Report the training accuracy and loss
	train_acc.append(100 * correct / total)
	train_loss.append(running_loss/total_step)
	print(f'\ntrain-loss: {np.mean(train_loss):.4f}, train-acc: {(100 * correct/total):.4f}')
	batch_loss = 0
	total_valid=0
	correct_valid=0
	# Suppress learning on the validation data
	with torch.no_grad():
		resnet.eval()
		for data_valid, target_valid in zip(X_test_dataloader,Y_test_dataloader):
			data_valid, target_valid = data_valid.to(device), target_valid.to(device)
			outputs_valid = resnet(data_valid.float())
			loss_valid = criterion(outputs_valid, target_valid)
			batch_loss += loss_valid.item()
			correct_valid += accuracy(outputs_valid,target_valid)
			total_valid += target_valid.size(0)
		# Report the validation accuracy and loss
		val_acc.append(100 * correct_valid/total_valid)
		val_loss.append(batch_loss/len(Y_test_dataloader))
		network_learned = batch_loss < valid_loss_min
		print(f'validation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_valid/total_valid):.4f}\n')

     
		if network_learned:
			valid_loss_min = batch_loss
			torch.save(resnet.state_dict(), 'resnet.pt')
			print('Improvement-Detected, save-model')
	resnet.train()











