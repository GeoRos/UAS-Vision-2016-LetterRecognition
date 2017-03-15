function convolvedFeatures = cnnConvolve(filterDim, numFilters, images, W, b)
%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  images - large images to convolve with, matrix in the form
%           images(r, c, image number)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numFilters)
%         b is of shape (numFilters,1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

numImages = size(images, 3);
imageDim = size(images, 1);
convDim = imageDim - filterDim + 1;

convolvedFeatures = zeros(convDim, convDim, numFilters, numImages);

%   Convolves every filter with every image to produce the 
%   (imageDim - filterDim + 1) x (imageDim - filterDim + 1) x numFeatures 
%   x numImages matrix convolvedFeatures, such that 
%   convolvedFeatures(imageRow, imageCol, featureNum, imageNum) is the
%   value of the convolved featureNum feature for the imageNum image over
%   the region (imageRow, imageCol) to 
%   (imageRow + filterDim - 1, imageCol + filterDim - 1)


for imageNum = 1:numImages
  for filterNum = 1:numFilters

    filter = squeeze(W(:,:,filterNum));

    % Flips the feature matrix because of the definition of convolution
    filter = rot90(squeeze(filter),2);

    im = squeeze(images(:, :, imageNum));
    
    % Convolves "filter" with "im", adding the result to convolvedImage
    convolvedImage = conv2(im, filter, 'valid');
    
    % Adds the bias unit
    convolvedImage = convolvedImage + b(filterNum);
    
    % Then, apply the activation function
    %Sigmoid
    convolvedImage = sigmf(convolvedImage, [1 0]);
    
    %ReLU
    %convolvedImage = max(convolvedImage, 0);
    
%     figure(1)
%     imshow(imresize(convolvedImage, [250 250]))
%     contOp = input(num2str(filterNum));
    
    convolvedFeatures(:, :, filterNum, imageNum) = convolvedImage;
  end
end


end

