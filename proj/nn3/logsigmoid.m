function [ logsig ] = logsigmoid( val, something )
%LOGSIGMOID The logistic function 1/(1+e^-t)
%   Well, self explanitory

logsig = 1 ./ (1 + (exp(-val)));

