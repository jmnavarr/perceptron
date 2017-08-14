function [ output, hidden ] = applyWeights( input, w1, w2 )
%APPLYWEIGHTS Applies the weights of the net to the input
%   Params:
%      input: column vector of length m
%      w1: input to hidden layer weights. mxn matrix.
%      w2: hidden to output layer weights. nxp matrix.
%   Output:
%      output: column vector of length p
%      hidden: output of the hidden layer
%
%   Special: if input is a matrix instead of a vector, then each column
%            is treated as an input vector. If there are q input vectors
%            then, for an input that's mxq, the output will be a matrix
%            of size pxq.

[len num] = size(input);

for i = 1:num
    hidden(:,i)=logsigmoid(w1*input(:,i));
    output(:,i)=logsigmoid(w2*hidden(:,i));
end
