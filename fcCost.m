function [ cost, grad, pred_prob, deltaFC] = fcCost( theta, ei, inputData, labels, numImages)
%FCCOST Simulates the operation of the Fully Connected layers
%   Forward Propagation of the Fully Connected layers
%   Calculates the gradients for the Fully Connected layers' parameters
%   Returns cost, gradients with respect to the Fully connected parameters
%   as an unrolled vector and network output probabilities. It also
%   backpropagates the errors through the first FC layer and returns it so
%   that it can be used by the convolutional layers.

%Inputs:
% theta     -   A vector that contains all of the parameters used by FC
%               layers
% ei        -   Structure that contains all of the characteristics of the
%               network
% inputData -   An unrolled vector of the convolved features
% labels    -   vector with length(numImages) that stores each the label of
%               each corresponding image
% numImages -   The number of images used

%Returns:
% cost      -   The cost of the network
% grad      -   A vector with length(grad) == length(theta). It contains
%               the gradients with respect to the FC weights and biases
% pred_prob -   A numClasses x numImages matrix containing the results
% deltaFC   -   The error backpropagated through the first layer of FC
%               weights. It is basically the error right after the pooling
%               layer.

%% reshape into network

stack = params2stack(theta, ei);        % A structure that contains the 
                                        % weights and biases of each layer.
                                        % i.e. stack{1}.W contains the
                                        % first layer of FC weights.
                                        
numHidden = numel(ei.layer_sizes) - 1;  % The number of hidden layers

hAct = cell(numHidden+1, 1);            % A CS List that contains the 
                                        % activations of every hidden layer
                                        % and the output layer
                                        
gradStack = cell(numHidden+1, 1);       % A cell structure with the same 
                                        % shape as stack that contains the
                                        % gradients with respect to each
                                        % corresponding parameter.
%% Forward Propagation through the Fully Connected layers

for i=1:numHidden+1 % Computes the neuron outputs for each hidden layer and 
                    % the output layer
    if(i==1)
        hAct{1}=bsxfun(@plus,stack{1}.W*inputData,stack{1}.b);
    else
        hAct{i}=bsxfun(@plus,stack{i}.W*hAct{i-1},stack{i}.b);
    end
    if(i<numHidden+1) % Computes the activations of each hidden layer
        switch ei.activation_fun
            case 'logistic'
                hAct{i}=sigmf(hAct{i}, [1 0]);
            case 'relu'
                hAct{i}=relu(hAct{i});
            case 'tanh'
                hAct{i}=tanh(hAct{i});
        end
    end
end

% Calculates the output probabilities
pred_prob=bsxfun(@rdivide,exp(hAct{end}),sum(exp(hAct{end}),1));


%% Cost Calculation
ind=sub2ind(size(pred_prob),labels',1:size(pred_prob,2));
cost=-sum(log(pred_prob(ind)))/numImages;

%% Error Backpropagation and Gradient Calculation

% Converts the vector of labels into a 2D numClasses x numImages matrix of
% 0s and 1s
labels_init=full(sparse([(1:3)';labels],1:numel(labels)+3,1));

labels_full=labels_init(:,4:end);

delta=cell(numHidden+1,1);  % A CS List with the same shape as hAct that 
                            % contains the errors for each layer
                            
delta{numHidden+1}=-(labels_full-pred_prob); % The output layer error

% Backpropagates the output layer error through the hidden layers
for i=numHidden:-1:1
    switch ei.activation_fun
        case 'logistic'
            delta{i}=stack{i+1}.W'*delta{i+1}.*(hAct{i}.*(1-hAct{i}));
        case 'relu'
            delta{i}=stack{i+1}.W'*delta{i+1}.*(hAct{i}>0);
        case 'tanh'
            delta{i}=stack{i+1}.W'*delta{i+1}.*(1-hAct{i}.^2);
    end
end

% Calculates the gradient w.r.t the FC parameters of each layer
for i=1:numHidden+1
    if(i==1)
        gradStack{i}.W=delta{i}*inputData'/numImages;
    else
        gradStack{i}.W=delta{i}*hAct{i-1}'/numImages;
    end
    gradStack{i}.b=sum(delta{i},2)/numImages;
end

%% Unroll FC Gradients into Vector
[grad] = stack2params(gradStack);

% Backpropagates the error through the first layer to be used later by the
% convolutional layers
deltaFC = stack{1}.W'*delta{1};
end


