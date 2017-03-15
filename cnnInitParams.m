function [theta, lengthFCParams] = cnnInitParams(imageDim, ei)
% Initialize parameters for a single layer convolutional neural
% network followed by multiple fully connected layers.
%                            
% Inputs:
%  imageDim      -  height/width of image
%  ei.filterDim  -  dimension of convolutional filter                            
%  ei.numFilters -  number of convolutional filters
%  ei.poolDim    -  dimension of pooling area
%
%
% Returns:
%  theta             -  unrolled parameter vector with initialized weights
%  lengthFCParams    -  fcLengthParams provides a point of reference for 
%                       the separation of the convolutional and fully 
%                       connected parameters in further steps.

%% Check if filter and pooling dimensions are sensible --------------------
assert(ei.filterDim < imageDim,'ei.filterDim must be less that imageDim');
outDim = imageDim - ei.filterDim + 1; % dimension of convolved image
assert(mod(outDim,ei.poolDim)==0,...
       'ei.poolDim must divide imageDim - ei.filterDim + 1');

%% Generate random weights and biases -------------------------------------
% Convolutional Layer
Wc = 3*randn(ei.filterDim,ei.filterDim,ei.numFilters);
bc = zeros(ei.numFilters, 1);

% Fully Connected Layers
% fcStack is a cell structure that contains separately the weights
% (fcStack.W) and the biases (fcStack.b) of the fully connected network.
% Each entry contains the weights and biases of the corresponding layer so
% that numel(fcStack) == numel(ei.layer_sizes)
% i.e. fcStack{1}.W contains the weights between the input layer and the
% first hidden layer.
fcStack = fcInitWeights(ei);

%% Convert weights and bias to the vector form ------------------
% This step will "unroll" (flatten and concatenate together) all 
% parameters into a vector, which can then be used during training.

[fcParams, lengthFCParams] = stack2params(fcStack);

theta = [Wc(:) ; bc(:) ; fcParams(:)];

end

