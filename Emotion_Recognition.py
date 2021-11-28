import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pdb
import pandas as pd
import seaborn as sns
from sklearn import svm, tree,linear_model
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, make_scorer, mean_squared_error,confusion_matrix
from sklearn.model_selection import train_test_split,cross_validate,GridSearchCV
from sklearn.naive_bayes import GaussianNB


def reassign_label(cluster_labels,y):
	'''
		Reassign with the most probable class label for each cluster in K-Means
		returns: dictionary of clusters assigned to each label
	'''

	ref_labels = {}

	# Loop through each cluster label, and reassign the class label
	for i in range(len(np.unique(cluster_labels))):
		index = np.where(cluster_labels == i,1,0)
		num = np.bincount(y[index==1]).argmax()
		ref_labels[i] = num

	return ref_labels

# Make a heatmap plot for the confusion matrix
def plot_ConfusionMatrix(y_actual,y_pred):
	pdb.set_trace()
	cm=confusion_matrix(y_actual,y_pred)
	cm_df=pd.DataFrame(cm,index=["Afraid","Angry","Disgusted","Happy","Neutral","Sad","Surprised"],
		columns=["Afraid","Angry","Disgusted","Happy","Neutral","Sad","Surprised"])
	fig, ax = plt.subplots(figsize=(12, 10))
	sns.heatmap(cm_df,annot=True)
	ax.set_title('Confusion Matrix', fontsize=20)
	ax.set_xlabel('Predicted Labels', fontsize=20)
	ax.set_ylabel('Actual Labels', fontsize=20)
	ax.set_xticklabels(["Afraid","Angry","Disgusted","Happy","Neutral","Sad","Surprised"],rotation=45)
	ax.set_yticklabels(["Afraid","Angry","Disgusted","Happy","Neutral","Sad","Surprised"],rotation=45)
	plt.savefig("CM.jpg")

# Train the specified model
def train_model(model_to_use,x,y,if_cv,if_plot,x_test=None,y_test=None):
	# Initialize the model
	if model_to_use=="Naive Bayes":
		model=GaussianNB()
	elif model_to_use=="Logistic Regression":
		model=linear_model.LogisticRegression(max_iter=5000)
	elif model_to_use=="Random Forest":
		model=RandomForestClassifier(n_estimators=25)
	elif model_to_use=="K-Means":
		model = MiniBatchKMeans(n_clusters = 22, batch_size=10)
	elif model_to_use=="SVM":
		param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['rbf','poly']}
		svc=svm.SVC(probability=True)
		model=GridSearchCV(svc,param_grid)

	# Model fitting
	if if_cv:
		# Perform 5-folds cross validation
		model_cv = cross_validate(model,x,y,cv=5,scoring={'Accuracy':make_scorer(accuracy_score),\
			"Mean squared error":make_scorer(mean_squared_error)},return_train_score=True)
		# Report accuracy and MSE metric results
		print("Average Training Accuracy ",np.mean(model_cv["train_Accuracy"]))
		print("Average Training MSE ",np.mean(model_cv["train_Mean squared error"]))
		print("Average Validation Accuracy ",np.mean(model_cv["test_Accuracy"]))
		print("Average Validation MSE ",np.mean(model_cv["test_Mean squared error"]))
	# Regular fitting
	else:
		model.fit(x,y)
		y_pred_train=model.predict(x)
		y_pred_valid=model.predict(x_test)
		# Label reassignment for SVM
		if model_to_use=="SVM":
			reference_labels_train = reassign_label(y_pred_train,y)
			reference_labels_train = reassign_label(y_pred_valid,y_test)
			class_labels_train = np.random.rand(len(y_pred_train))
			class_labels_valid = np.random.rand(len(y_pred_valid))
			for i in range(len(class_labels_train)):
				class_labels_train[i] = reference_labels_train[y_pred_train[i]]
			for i in range(len(class_labels_valid)):
				class_labels_valid[i] = reference_labels_valid[y_pred_valid[i]]
			# Report metric result
			print("Average Training Accuracy ",sum(class_labels_train==y)/len(y))
			print("Average Validation Accuracy ",sum(class_labels_valid==y_test)/len(y_test))
		else:
			print("Average Training Accuracy ",sum(y_pred_train==y)/len(y))
			print("Average Validation Accuracy ",sum(y_pred_valid==y_test)/len(y_test))

		# Plot model result
		if if_plot:
			if model_to_use=="Naive Bayes":
				plot_ConfusionMatrix(y_test,y_pred_valid)
			elif model_to_use=="Random Forest":
				fig, axes = plt.subplots(figsize =(6,6))
				tree.plot_tree(model.estimators_[0])
				plt.savefig("RF.jpg")

def main(num_folder,model_to_use,if_augment,if_cv,if_plot):

	label_dict={"AF":0,"AN":1,"DI":2,"HA":3,"NE":4,"SA":5,"SU":6}
	X_data, Y_label=[],[]
	image_dir="/Users/hanqingyao/Downloads/KDEF_masked_all"
	image_subdirs=[x[0] for x in os.walk(image_dir)][1:]

	# Loop through each folder to construct the feature and label data
	for subdir in image_subdirs[:num_folder]:
		files = os.walk(subdir).__next__()[2]
		for file in files:
			if (file.find("surgical_blue")!=-1)|(file.find("surgical_green")!=-1):
				continue
			im=cv2.imread(os.path.join(subdir,file))

			# Data augmentation mode
			if if_augment:
				Y_label+=[label_dict[file[4:6]]]*7
				# Crop the lower 3/7 portion and resize the image
				crop_im=im[:im.shape[0]*4//7,:]
				crop_im=cv2.resize(crop_im,(64,64))/255
				im=cv2.resize(im,(64,64))
				# Add rotated, flipped, cropped together with the original images to the feature data
				X_data+=[im.flatten()/255]+[cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE).flatten()/255]+\
				[cv2.rotate(im,cv2.ROTATE_180).flatten()/255]+\
				[cv2.rotate(im,cv2.ROTATE_90_COUNTERCLOCKWISE).flatten()/255]+\
				[cv2.flip(im,0).flatten()/255]+\
				[cv2.flip(im,1).flatten()/255]+[crop_im.flatten()/255]
			# Regular mode
			else:
				Y_label.append(label_dict[file[4:6]])
				im=cv2.resize(im,(64,64))
				X_data.append(im.flatten()/255)
	
	X_data = np.stack(X_data)
	Y_label = np.stack(Y_label)

	# Cross validation mode
	if if_cv:
		# Shuffle the data
		np.random.seed(0)
		shuffle_index=np.arange(X_data.shape[0])
		np.random.shuffle(shuffle_index)
		X_data=X_data[shuffle_index]
		Y_label=Y_label[shuffle_index]
		train_model(model_to_use,X_data,Y_label,if_cv,if_plot)
	# Regular mode
	else:
		# 80% for training, 20% for validation
		X_train,X_test,Y_train,Y_test=train_test_split(X_data,Y_label,test_size=0.2)
		train_model(model_to_use,X_train,Y_train,if_cv,if_plot,X_test,Y_test)

# Arguments
# num_folder: number of image folders to include in training and testing
# model_to_use: select a model from Naive Bayes, Logistic Regression, Random Forest, K-Means, SVM
# if_augment: indicator of whether to use data augmentation(transform images)
# if_cv: indicator of whether to perform cross validation or one 80% train vs 20% validation model fitting
# if_plot: whether to create a customized plot based on the model
if __name__ == '__main__':
    main(num_folder=100,model_to_use="Naive Bayes",if_augment=False, if_cv=False, if_plot=True)

