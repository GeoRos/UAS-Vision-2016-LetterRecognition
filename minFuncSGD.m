function [opttheta] = minFuncSGD(funObj,theta,data,labels,options)
% Runs stochastic gradient descent with momentum to optimize the
% parameters for the given objective.
%
% Parameters:
%  funObj     -  function handle which accepts as input theta,
%                data, labels and returns cost and gradient w.r.t
%                to theta.
%  theta      -  unrolled parameter vector
%  data       -  stores data in m x n x numExamples tensor
%  labels     -  corresponding labels in numExamples x 1 vector
%  options    -  struct to store specific options for optimization
%
% Returns:
%  opttheta   -  optimized parameter vector
%
% Options (* required)
%  epochs*     - number of epochs through data
%  alpha*      - initial learning rate
%  minibatch*  - size of minibatch
%  momentum    - momentum constant, default to 0.9


%%======================================================================
%% Setup
assert(all(isfield(options,{'epochs','alpha','minibatch'})),...
        'Some options not defined');
if ~isfield(options,'momentum')
    options.momentum = 0.9;
end
epochs = options.epochs;
alpha = options.alpha;
minibatch = options.minibatch;
m = length(labels); % training set size
% Setup for momentum
mom = 0.2;
momIncrease = 20;
optCost = 100;
velocity = zeros(size(theta));

%%======================================================================
%% SGD loop
it = 0;
prevCost = 0;
totalTime = 0;
disp(m-minibatch+1)

for e = 1:epochs
    
    for repeats = 1:10
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(m);
    
    for s=1:minibatch:(m-minibatch+1)
        tic;
        it = it + 1;

        % increase momentum after momIncrease iterations
        if it == 20
            mom = options.momentum;
        end;

        % get next randomly selected minibatch
        mb_data = data(:,:,rp(s:s+minibatch-1));
        mb_labels = labels(rp(s:s+minibatch-1));

        % evaluate the objective function on the next minibatch
        [cost,grad] = funObj(theta,mb_data,mb_labels);
        
        velocity=mom*velocity+alpha*grad;
        theta=theta-velocity;
        
        fprintf('\nEpoch %d: Cost on iteration %d is %f\n',e,it,cost);
        disp(['Iteration time: ', num2str(toc), 'sec'])
        totalTime = totalTime + toc;
        disp(['Total time: ', num2str(totalTime), 'sec, ',...
            num2str(totalTime/60), 'min, ', num2str(totalTime/3600), 'h'])
        estimatedTime = totalTime/it*(m-minibatch+1)/minibatch*10*epochs/3600;
        disp(['Estimated Time for optimization: ', num2str(estimatedTime), ' hours'])
        
        if cost<optCost
            optCost = cost;
            bestCostTheta = theta;
        end
    end;

    end
    % aneal learning rate by factor of two after each epoch
    alpha = alpha/2.0;

end

opttheta = bestCostTheta;

end