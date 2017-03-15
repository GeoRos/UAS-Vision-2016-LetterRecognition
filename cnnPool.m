function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);

pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

%   Pools the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   (convolvedDim/poolDim) x (convolvedDim/poolDim) x numFeatures 
%   x numImages matrix pooledFeatures, such that
%   pooledFeatures(poolRow, poolCol, featureNum, imageNum) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region. 

for x = 1:size(pooledFeatures,1)
    for y = 1:size(pooledFeatures,1)
        
        %Uncomment for MeanPooling
        pooledFeatures(x, y, :, :) = ...
            mean(mean(convolvedFeatures(((x-1)*poolDim)+1:(x*poolDim),...
            ((y-1)*poolDim)+1:(y*poolDim), :, :)));
        
        %Uncomment for MaxPooling
%         pooledFeatures(x, y, :, :) = ...
%             max(max(convolvedFeatures(((x-1)*poolDim)+1:(x*poolDim),...
%             ((y-1)*poolDim)+1:(y*poolDim), :, :)));
    end
end


end

