import matplotlib.pyplot as plt

from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from mnist import MNIST
import numpy as np
import pickle

#digits = datasets.load_digits()
#print(digits.data)

data = MNIST('./python-mnist/data/')

img_train, labels_train = data.load_training()
train_img = np.array(img_train)
train_labels = np.array(labels_train)

img_test, labels_test = data.load_testing()
test_img = np.array(img_test)
test_labels = np.array(labels_test)


#Features
X = train_img

#Labels
y = train_labels

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.1)

clf = RandomForestClassifier(n_estimators=100, n_jobs=10,)
clf.fit(X_train,y_train)

with open('MNIST_RFC.pickle','wb') as f:
	pickle.dump(clf, f)

pickle_in = open('MNIST_RFC.pickle','rb')
clf = pickle.load(pickle_in)

acc = clf.score(X_test,y_test)
print('RFC Score: ',acc)

#---------------------- EOC --------------------------#
