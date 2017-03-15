# UAS-Vision-2016
Convolutional Neural Net for Letter-On-Target Recognition (MATLAB)

This code trains a Convolutional Neural Network to be able to identify what letter is written on the target as described by the IMechE Competition's Target Description (https://www.imeche.org/get-involved/young-members-network/auasc). The network is currently trained to be able to identify the letters [A,C,E] with an accuracy of 99.8% with grayscale input image, a single convolutional layer and a single fully-connected hidden layer at the end. There also exist samples for the letters [H,L,O,T] for further training.

The network is based on the Convolution Neural Network produced after finishing the Stanford UFLDL Tutorial (http://ufldl.stanford.edu/tutorial/), although it has been changed a lot.

INSTRUCTIONS
Begin the program by running the cnnTrain script. The network characteristics can be changed from the same script. The optimumTheta.mat file contains the optimum parameters required to achieve 99.8% accuracy. In order for these parameters to work the network needs to have the following characteristics:

image dimensions  : 30
number of classes : 3
filter dimensions : 9
number of filter  : 30
pooling dimensions: 2
fully-connected layer sizes: [128, (number of classes)]

If training of a new network is required, make sure to first copy the current optimum parameters to another directory since they will be overwritten as soon as training finishes
