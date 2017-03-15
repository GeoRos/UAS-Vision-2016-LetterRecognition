function [ trainData, trainLabels, testData, testLabels ] = loadData( )
%LOADDATA Summary of this function goes here
%   Detailed explanation goes here

% imageData = zeros(60,60,50000);
% imageLabels = zeros(50000,1);
load('LetterASamples.mat');
ASize = size(letterACollection,3);
load('LetterCSamples.mat');
CSize = size(letterCCollection,3);
load('LetterESamples.mat');
ESize = size(letterECollection,3);
% load('LetterLSamples.mat');
% LSize = size(letterLCollection,3);
% load('LetterOSamples.mat');
% OSize = size(letterOCollection,3);
% load('LetterTSamples.mat');
% TSize = size(letterTCollection,3);

imageData = zeros(30,30,ASize+CSize+ESize);
imageData(:,:,1:ASize) = im2double(letterACollection);
imageLabels = ones(ASize,1);
imageData(:,:,ASize+1:      ASize+CSize) = im2double(letterCCollection);
imageLabels  (ASize+1:      ASize+CSize) = 2;
imageData(:,:,ASize+CSize+1:ASize+CSize+ESize) = im2double(letterECollection);
imageLabels  (ASize+CSize+1:ASize+CSize+ESize) = 3;
% imageData(:,:,ASize+CSize+ESize+1:            ASize+CSize+ESize+LSize) = im2double(letterLCollection);
% imageLabels  (ASize+CSize+ESize+1:            ASize+CSize+ESize+LSize) = 4;
% imageData(:,:,ASize+CSize+ESize+LSize+1:      ASize+CSize+ESize+LSize+OSize) = im2double(letterOCollection);
% imageLabels  (ASize+CSize+ESize+LSize+1:      ASize+CSize+ESize+LSize+OSize) = 5;
% imageData(:,:,ASize+CSize+ESize+LSize+OSize+1:ASize+CSize+ESize+LSize+OSize+TSize) = im2double(letterTCollection);
% imageLabels  (ASize+CSize+ESize+LSize+OSize+1:ASize+CSize+ESize+LSize+OSize+TSize) = 6;
% % imageLabels = zeros(2*ASize+CSize+ESize+nonTargetSizeData,1);
% imageLabels(1:2*ASize) = 1;
%ACELOT
disp(size(imageLabels))

%Random permutation
I = randperm(size(imageData,3));
imageData = imageData(:,:,I);
imageLabels = imageLabels(I);

trainLabels = imageLabels(1:end-2000);
testLabels = imageLabels(end-1999:end);
trainData = imageData(:,:,1:end-2000);
testData = imageData(:,:,end-1999:end);

% contOp = [];
% while isempty(contOp)
%     randomSamp = ceil(rand*40000);
%     imshow(imageData(:,:,randomSamp))
%     contOp = input(num2str(imageLabels(randomSamp)));
% end