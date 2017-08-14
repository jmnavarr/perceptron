function [ nwH, nwO, nerror, noutput, nhidden, ne ] = updateWeights( input, target, wH, wO, lrate, output, hidden, e)
%UPDATEWEIGHTS Updates the weights of a neural net.
%   Detailed explanation goes here


% current output of weights
if nargin < 6
    [output hidden] = applyWeights(input, wH, wO);
    e = target - output;
end

% get number of inputs
[len ninput] = size(input);

% iterate and update weights across all inputs
for i = 1:ninput
    % derivatives of the output and hidden layers
    dO = dlogsigmoid(output(:,i));
    dH = dlogsigmoid(hidden(:,i));

    % error slope at each output/hidden node
    eO = dO .* e(:,i);
    eH = dH .* (wO' * eO);

    % update weights
    wO = wO + lrate * eO * hidden(:,i)';
    wH = wH + lrate * eH * input(:,i)';

    % get new outputs with new weights
    [output(:,i) hidden(:,i)] = applyWeights(input(:,i), wH, wO);
end

e = target - output;

nwH = wH;
nwO = wO;
nerror = .5*mean(mean(e.*e));
noutput = output;
nhidden = hidden;
ne = e;
