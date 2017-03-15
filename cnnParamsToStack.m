function [Wc, bc, thetaFC] = cnnParamsToStack(theta,filterDim,...
                                 numFilters,lengthParams)
% Converts unrolled parameters for a single layer convolutional neural
% network followed by a softmax layer into structured weight
% tensors/matrices and corresponding biases
%                            
% Parameters:
%  theta      -  unrolled parameter vectore
%  filterDim  -  dimension of convolutional filter                            
%  numFilters -  number of convolutional filters
%  numClasses -  number of classes to predict
%
%
% Returns:
%  Wc      -  filterDim x filterDim x numFilters parameter matrix
%  bc      -  bias for convolution layer of size numFilters x 1
%  thetaFc -  unrolled parameter vector of Fully Connected parameters


%% Reshape theta
thetaFC = theta(end-lengthParams+1:end);

indS = 1;
indE = filterDim^2*numFilters;
Wc = reshape(theta(indS:indE),filterDim,filterDim,numFilters);

indS = indE+1;
indE = indE+numFilters;
bc = theta(indS:indE);



end