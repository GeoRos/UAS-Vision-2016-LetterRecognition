function [ stack ] = fcInitWeights( ei )
%FCINITWEIGHTS Random weight structures for a fully connected network 
%architecture
%   ei describes the network via the fields layerSizes, inputDim, and 
%   outputDim 
%   
%   This uses Xavier's weight initialization tricks for better backprop

%% initialize hidden layers
stack = cell(1, numel(ei.layer_sizes));
for l = 1 : numel(ei.layer_sizes)
    if l > 1
        prev_size = ei.layer_sizes(l-1);
    else
        prev_size = ei.input_dim;
    end;
    cur_size = ei.layer_sizes(l);
    % Xaxier's scaling factor
    s = sqrt(6) / sqrt(prev_size + cur_size);
    stack{l}.W = rand(cur_size, prev_size)*2*s - s;
    stack{l}.b = zeros(cur_size, 1);
end
