import cv2
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf



X_cnn_data, Y_cnn_label=[],[]
label_dict={"AF":0,"AN":1,"DI":2,"HA":3,"NE":4,"SA":5,"SU":6}

image_dir="/Users/hanqingyao/Downloads/KDEF_masked_all"
image_subdirs=[x[0] for x in os.walk(image_dir)][1:]

for subdir in image_subdirs[:100]:
	files = os.walk(subdir).__next__()[2]
	for file in files:
		if (file.find("surgical_blue")!=-1)|(file.find("surgical_green")!=-1):
			continue
		im=cv2.imread(os.path.join(subdir,file))
		y=np.zeros(7)
		y[label_dict[file[4:6]]]=1
		Y_cnn_label.append(y)

		im=cv2.resize(im,(64,64))

		X_cnn_data.append(im/255)


X_cnn_data = np.stack(X_cnn_data)

Y_cnn_label = np.stack(Y_cnn_label)
X_cnn_train,X_cnn_test,Y_cnn_train,Y_cnn_test=train_test_split(X_cnn_data,Y_cnn_label,test_size=0.2)


model = Sequential()


# Construct the convolutional neutral network
# convolutional layer
model.add(Conv2D(250, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(64,64,3)))



# convolutional layer
model.add(Conv2D(125, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))



model.add(Conv2D(75, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))



# flatten output of conv
model.add(Flatten())



# hidden layer
model.add(Dense(500, activation='sigmoid'))
model.add(Dropout(0.4))

# output layer
model.add(Dense(7, activation='relu'))

# compiling the sequential model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')



# training the model for 10 epochs with batch size 60
model.fit(X_cnn_train, Y_cnn_train, batch_size=60, epochs=10, validation_data=(X_cnn_test, Y_cnn_test))




