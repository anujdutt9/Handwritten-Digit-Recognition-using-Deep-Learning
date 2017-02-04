import numpy as np
import argparse
import cv2
from cnn import CNN
from keras.utils import np_utils
from keras import optimizers
from sklearn.datasets import fetch_mldata
from sklearn import cross_validation


# Parse the Arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save_model", type=int, default=-1)
ap.add_argument("-l", "--load_model", type=int, default=-1)
ap.add_argument("-w", "--saved_weights", type=str)
args = vars(ap.parse_args())

# Read/Download MNIST Dataset
print('Loading MNIST Dataset...')
dataset = fetch_mldata('MNIST Original')

# Read the MNIST data as array of 784 pixels and convert to 28x28 image matrix 
mnist_data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
mnist_data = data[:, np.newaxis, :, :]

# Divide data into testing and training sets.
train_img, test_img, train_labels, test_labels = cross_validation.train_test_split(mnist_data/255.0, dataset.target.astype("int"), test_size=0.1)

# Now each image rows and columns are of 28x28 matrix type.
img_rows,img_columns = 28,28

# Transform training and testing data to 10 classes in range [0,classes] ; num. of classes = 0 to 9 = 10 classes
total_classes = 10			# 0 to 9 labels
train_labels = np_utils.to_categorical(train_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)

# Defing and compile the SGD optimizer and CNN model
print('\n Compiling model...')
sgd = optimisers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
clf = CNN(width=img_rows, height=img_columns, depth=1, total_classes=total_classes, weightsPath=args["weights"])
clf.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Initially train and test the model; If weight saved already, load the weights using arguments.
b_size = 128		# Batch size
num_epoch = 20		# Number of epochs
verb = 1			# Verbose

# If weights saved and argument load_model; Load the pre-trained model.
if args["load_model"] < 0:
	print('\nTraining the Model...')
	clf.fit(trainData, trainLabels, batch_size=b_size, nb_epoch=num_epoch,verbose=verb)
	
	# Evaluate accuracy and loss function of test data
	print('Evaluating Accuracy and Loss Function...')
	loss, accuracy = clf.evaluate(test_img, test_labels, batch_size=128, verbose=1)
	print('Accuracy of Model: {:.2f}%'.format(accuracy * 100))

	
# Save the pre-trained model.
if args["save_model"] > 0:
	print('Saving weights to file...')
	clf.save_weights(args["weights"], overwrite=True)

	
# Show the images using OpenCV and making random selections.
for num in np.random.choice(np.arange(0, len(test_labels)), size=(5,)):
	# Predict the label of digit using CNN.
	probs = clf.predict(test_img[np.newaxis, i])
	prediction = probs.argmax(axis=1)

	# Resize the Image to 100x100 from 28x28 for better view.
	image = (test_img[num][0] * 255).astype("uint8")
	image = cv2.merge([image] * 3)
	image = cv2.resize(image, (100, 100), interpolation=cv2.INTER_LINEAR)
	cv2.putText(image, str(prediction[0]), (5, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

	# Show and print the Actual Image and Predicted Label Value
	print('Predicted Label: {}, Actual Value: {}'.format(prediction[0],np.argmax(testLabels[num])))
	cv2.imshow('Digits', image)
	cv2.waitKey(0)
	
#---------------------- EOC ---------------------
