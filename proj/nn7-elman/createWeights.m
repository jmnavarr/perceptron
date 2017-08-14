function [ w1, w2 ] = createWeights( input, target, numhidden )
%CREATEWEIGHTS Summary of this function goes here
%   Detailed explanation goes here

[ninput,burn] = size(input);
[noutput,burn] = size(target);

a=0.3;                        % define the range of random variables
b=-0.3;

w1=a + (b-a) *rand(numhidden,ninput);     % Weights between Input and Hidden Neurons
w2=a + (b-a) *rand(noutput,numhidden);    % Weights between Hidden and Output Neurons
