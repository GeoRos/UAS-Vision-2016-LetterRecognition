# UAS-Vision-2016
Convolutional Neural Net for Letter-On-Target Recognition (MATLAB)

This code trains a Convolutional Neural Network to be able to identify what letter is written on the target as described by the IMechE Competition's Target Description (https://www.imeche.org/get-involved/young-members-network/auasc). The network is currently trained to be able to identify the letters [A,C,E] with an accuracy of 99.8% with input images of size 30x30, a single convolutional layer with 30 filters of size 9x9 and a single fully-connected hidden layer of size 128 at the end. There also exist samples for the letters [H,L,O,T] for further training.

The network is based on the Convolution Neural Network produced after finishing the Stanford UFLDL Tutorial (http://ufldl.stanford.edu/tutorial/), although it has been changed a lot.
