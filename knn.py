import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from mnist import MNIST

import numpy as np
import pickle

#digits = datasets.load_digits()
#print(digits.data)

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

print('\nPreparing Classifier Training and Testing Data...')
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.1)


print('\nKNN Classifier with n_neighbors = 5, algorithm = auto, n_jobs = 10')
print('\nPickling the Classifier for Future Use...')
clf = KNeighborsClassifier(n_neighbors=5,algorithm='auto',n_jobs=10)
clf.fit(X_train,y_train)

with open('MNIST_KNN.pickle','wb') as f:
	pickle.dump(clf, f)

pickle_in = open('MNIST_KNN.pickle','rb')
clf = pickle.load(pickle_in)

print('\nCalculating Accuracy of trained Classifier...')
acc = clf.score(X_test,y_test)

print('\nMaking Predictions on Testing Data...')
pred = clf.predict(test_img)

print('\nCalculating Accuracy of Predictions...')
accuracy = accuracy_score(test_labels, pred)

print('\nKNN Trained Classifier Accuracy: ',acc)
print('\nPredicted Values: ',pred)
print('\nAccuracy of Classifier on Test Images: ',accuracy)

#------------------------- EOC -----------------------------
