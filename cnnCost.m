function [cost, grad, preds] = cnnCost(theta,images,labels,numClasses,...
                                ei,fcLengthParams,pred)
% Calculate cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta            -  unrolled parameter vector
%  images           -  stores images in imageDim x imageDim x numImages
%                      array
%  labels           -  vector with length(numImages) that stores each
%                      the label of each corresponding image
%  numClasses       -  number of classes to predict
%  ei.filterDim     -  dimension of convolutional filter                            
%  ei.numFilters    -  number of convolutional filters
%  ei.poolDim       -  dimension of pooling area
%  pred             -  boolean. When true, only forward propagation is
%                      carried out and the predictions are returned
%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)

if ~exist('pred','var')
    pred = false;
end

imageDim = size(images,1);	% height/width of image
numImages = size(images,3); % number of images
%% Reshape parameters and setup gradient matrices for conv layer

% Wc is ei.filterDim x ei.filterDim x ei.numFilters parameter matrix
% bc is the corresponding bias
% thetaFC contains the parameters of the fully connected layers still in
%   vector form, to be converted later.

[Wc, bc, thetaFC] = cnnParamsToStack(theta,ei.filterDim,...
    ei.numFilters,fcLengthParams);

% Same sizes as Wc,bc. Used to hold gradient w.r.t conv parameters.
Wc_grad = zeros(size(Wc));
bc_grad = zeros(size(bc));

%%======================================================================
%% Forward Propagation through Convolutional Layer
%  For each image and each filter, convolves the image with the filter, 
%  adds the bias and applies the activation function.  Then subsamples the 
%  convolved activations with mean/max pooling.  Stores the results of the
%  convolution in activations and the results of the pooling in
%  activationsPooled. 

convDim = imageDim-ei.filterDim+1; % dimension of convolved output
outputDim = convDim/ei.poolDim;    % dimension of subsampled output

% convDim x convDim x ei.numFilters x numImages tensor for storing
% activations
activations = zeros(convDim,convDim,ei.numFilters,numImages);
activations = activations + cnnConvolve(ei.filterDim, ei.numFilters,...
    images, Wc, bc);

% outputDim x outputDim x ei.numFilters x numImages tensor for storing
% subsampled activations
activationsPooled = zeros(outputDim,outputDim,ei.numFilters,numImages);
activationsPooled = activationsPooled + cnnPool(ei.poolDim, activations);

% Reshapes activations into 2-d matrix, hiddenSize x numImages for Fully
% Connected Layers
activationsPooledReshaped = reshape(activationsPooled,[],numImages);
%% Fully Connected Layers
%  fcCost is responsible for every task carried out in the fully connected
%  layers.
%  It forward propagates the pooled activations calculated above into the
%  fully connected layers. For convenience activationsPooled was reshaped
%  into a hiddenSize x numImages matrix. 
%  Returns the results in probs. Calculates cost and gradients for Fully 
%  Connected layers' parameters. Also returns the error just after the 
%  pooling layer.

[ cost, FC_grad, probs, delta_pooled] = fcCost( thetaFC, ei, ...
    activationsPooledReshaped, labels, numImages);

% Replaces the highest probability of each image with 1 and the rest with 0
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end

%%======================================================================
%% Backpropagation through Convolutional Layer
%  Backpropagates the fully connected errors through the convolutional and
%  subsampling layers. The errors will be used later to calculate the
%  gradients

%  The errors are reshaped into 
%  outputDim x outputDim x ei.filterNum x numImages
delta_pooled=reshape(delta_pooled,[],outputDim,ei.numFilters,numImages);

% outputDim*ei.poolDim x outputDim*ei.poolDim x ei.numFilters x numImages 
% tensor for storing the unpooled error
delta_unpooled=zeros(outputDim*ei.poolDim,outputDim*ei.poolDim,...
    ei.numFilters,numImages);

% Uncomment for MaxPooling
% activationsUnpooled=zeros(outputDim*ei.poolDim,outputDim*ei.poolDim,ei.numFilters,numImages);

for imageNum=1:numImages
    for filterNum=1:ei.numFilters
        %Uncomment for MaxPooling
%         activationsUnpooled(:,:,filterNum,imageNum) = ...
%             kron(activationsPooled(:,:,filterNum,imageNum),ones(ei.poolDim,ei.poolDim));
%         delta_unpooled(:,:,filterNum,imageNum)= ...
%             kron(delta_pooled(:,:,filterNum,imageNum),ones(ei.poolDim,ei.poolDim)); % upsampling
        
        %Uncomment for MeanPooling
        delta_unpooled(:,:,filterNum,imageNum)= (1/ei.poolDim^2).*...
            kron(delta_pooled(:,:,filterNum,imageNum),ones(ei.poolDim,ei.poolDim)); % upsampling
    end
end
% Uncomment for MaxPooling
% delta_unpooled(activationsUnpooled~=activations)=0;

% Activation function is applied
% Uncomment for Sigmoid
delta_conv=delta_unpooled.*activations.*(1-activations);

% Uncomment for ReLU
% delta_conv=delta_unpooled.*(activations>0);

%%======================================================================
%% Convolutional Gradient Calculation
%  After backpropagating the errors above, they are used to calculate the
%  gradient with respect to the convolutional parameters.

% Calculates gradients w.r.t each weight
for filterNum=1:ei.numFilters
    for imageNum=1:numImages
        Wc_grad(:,:,filterNum,:)=Wc_grad(:,:,filterNum,:)+...
            conv2(images(:,:,imageNum),rot90(delta_conv(:,:,filterNum,imageNum),2),'valid');
    end
    Wc_grad(:,:,filterNum,:)=Wc_grad(:,:,filterNum,:)/numImages;
end

% Calculates gradients w.r.t each bias
for filterNum = 1 : ei.numFilters
    bc_grad(filterNum) = sum(sum(sum(delta_conv(:,:,filterNum,:))))...
        /numImages;
end

% Unrolls convolutional parameters and combines them with the fully
% connected parameters into one vector
grad = [Wc_grad(:) ; bc_grad(:) ; FC_grad];

end