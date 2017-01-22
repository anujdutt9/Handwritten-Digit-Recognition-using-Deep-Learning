import matplotlib.pyplot as plt

from sklearn import cross_validation, svm, preprocessing
from sklearn.metrics import accuracy_score
from mnist import MNIST

import numpy as np
import pickle


print('\nLoading MNIST Data...')
data = MNIST('./python-mnist/data/')

print('\nLoading Training Data...')
img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)

print('\nLoading Testing Data...')
img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)


#Features
X = train_img

#Labels
y = train_labels

# Prepare Classifier Training and Testing Data
print('\nPreparing Classifier Training and Testing Data...')
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.1)


# Pickle the Classifier for Future Use
print('\nSVM Classifier with gamma = 0.1; Kernel = polynomial')
print('\nPickling the Classifier for Future Use...')
clf = svm.SVC(gamma=0.1, kernel='poly')
clf.fit(X_train,y_train)

with open('MNIST_SVM.pickle','wb') as f:
	pickle.dump(clf, f)

pickle_in = open('MNIST_SVM.pickle','rb')
clf = pickle.load(pickle_in)

print('\nCalculating Accuracy of trained Classifier...')
acc = clf.score(X_test,y_test)

print('\nMaking Predictions on Testing Data...')
pred = clf.predict(test_img)

print('\nCalculating Accuracy of Predictions...')
accuracy = accuracy_score(test_labels, pred)

print('\nSVM Trained Classifier Accuracy: ',acc)
print('\nPredicted Values: ',pred)
print('\nAccuracy of Classifier on Test Images: ',accuracy)

#for i in np.random.choice(np.arrange(0,len(test_img)), size(10,)):
#	img = (test_img[0]*255).reshape((28,28)).astype('uint8')
#	print('Actual digit is {0}, predicted {1}'.format(test_img[i],pred[0])
#	cv2.imshow('cat',img)
#	cv2.waitKey(0)

#---------------------- EOC ---------------------#

