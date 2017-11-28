# Handwritten Digit Recognition using Machine Learning and Deep Learning

## Published Paper 

[IJARCET-VOL-6-ISSUE-7-990-997](http://ijarcet.org/wp-content/uploads/IJARCET-VOL-6-ISSUE-7-990-997.pdf)

# Requirements

* Python 3.5 +
* Scikit-Learn (latest version)
* Numpy (+ mkl for Windows)
* Matplotlib

# Usage

Download the four MNIST dataset files from this link:

```
http://yann.lecun.com/exdb/mnist/
```

Unzip and place the files in the dataset folder inside the MNIST_Dataset_Loader folder under each ML Algorithm folder i.e :

```
KNN
|_ MNIST_Dataset_Loader
   |_ dataset
      |_ train-images-idx3-ubyte
      |_ train-labels-idx1-ubyte
      |_ t10k-images-idx3-ubyte
      |_ t10k-labels-idx1-ubyte
```

Do this for SVM and RFC folders and you should be good to go.

## Accuracy using Machine Learning Algorithms:

i)	 K Nearest Neighbors: 96.67%

ii)	 SVM:	97.91%

iii) Random Forest Classifier:	96.82%


## Accuracy using Deep Neural Networks:

i)	Three Layer Convolutional Neural Network using Tensorflow:	99.70%

ii)	Three Layer Convolutional Neural Network using Keras and Theano: 98.75%

**All code written in Python 3.5. Code executed on Intel Xeon Processor / AWS EC2 Server.**

## Video Link:
```
https://www.youtube.com/watch?v=7kpYpmw5FfE
```

## Test Images Classification Output:

![Output a1](Outputs/output.png?raw=true "Output a1")       
