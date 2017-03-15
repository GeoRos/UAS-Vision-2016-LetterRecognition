
%% Single Layer Convolutional Neural Network Training and Testing

% This file contains the characteristics of the neural network and it is
% where each process begins. The neural network contains only one
% convolutional layer but it can contain multiple fully-connected layers

% PROCESSES
% Training ---------------------------------------------------------------
% Trains the full network, using Stochastic Gradient Descend.

% Testing ----------------------------------------------------------------
% Tests the optimised network using a small proportion of the input data
% and labels that was not included in the training.

%%========================================================================
%% Specificy Network Characteristics and Load Training and Testing Data

% USER INPUT =============================================================
finalTest = true; % True : Test using optimised parameters without training
                  % False: Initialise new parameters, train and test

% Load touch sensor data--------------------------------------------------
imageDim = 30;  % Dimensions of input images
numClasses = 3; % Number of classes
[images, labels, testImages, testLabels] = loadData;


% Experimental Information (Network Characteristics)----------------------
% Convolutional Layer
ei.filterDim = 9;      % Filter size for conv layer
ei.numFilters = 30;    % Number of filters for conv layer
ei.poolDim = 2;         % Pooling dimension, 
                        % (should divide imageDim-ei.filterDim+1)
                        
% Fully Connected Layers
convOutputSize =((imageDim - ei.filterDim + 1)/ei.poolDim)^2*ei.numFilters;
ei.input_dim = convOutputSize;  % dimension of input features
ei.output_dim = numClasses;     % number of output classes
ei.layer_sizes = [128 ei.output_dim]; % sizes of all hidden layers 
                                            % and the output layer
ei.activation_fun = 'logistic'; % activation function to be used

% USER INPUT(END)==========================================================

% Load Parameters --------------------------------------------------------
% Initialises new parameters if finalTest is False
if ~finalTest
    fprintf('Initialising random parameters: ');
    [theta, lengthFCParams] = cnnInitParams(imageDim,ei);
    fprintf('Done\n');
else
% Uses optimised parameters if finalTest is True
    fprintf('Loading optimum parameters: ');
    load ('optimumTheta.mat');
    fprintf('Done\n');
end

%%======================================================================
%% Training
%  Trains the full network, using Stochastic Gradient Descend.
if finalTest
    training=false; % Change to true to initiate training
else
    training=true; % Change to true to initiate training
end

if training
    disp('Training')
    options.epochs = 5;         % Number of training epochs
    options.minibatch = 500;    % Training minibatch size
    options.alpha = 1e-1;       % Learning Rate
    options.momentum = .95;     % Training Momentum

    % Start Training
    [opttheta] = minFuncSGD(@(x,y,z) cnnCost(x,y,z,numClasses,ei,lengthFCParams),theta,images,labels,options);
end
%%======================================================================
%% Testing
% Tests the optimised network using a small proportion of the input data
% and labels that was not included in the training.
disp ('Testing')

[~,~,preds]=cnnCost(opttheta,testImages,testLabels,numClasses,...
                ei,lengthFCParams,true);

acc = sum(preds==testLabels)/length(preds);

fprintf('Accuracy is %f%%\n',acc*100);

keepTesting =[];
prompt = 'Image ID';
while isempty(keepTesting)
    randomImage = ceil(rand*size(testImages,3));
    imshow(testImages(:,:,randomImage))
    [~,~,preds]=cnnCost(opttheta,testImages(:,:,randomImage),testLabels(randomImage),numClasses,...
                ei,lengthFCParams,true);
    disp([num2str(round(preds)), ' ', num2str(testLabels(randomImage))])
    keepTesting = input([prompt, ' ',num2str(randomImage),': ']);
end


%Save optimised parameters
save('optimumTheta', 'opttheta', 'lengthFCParams')

