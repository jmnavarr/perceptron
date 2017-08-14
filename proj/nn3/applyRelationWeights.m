function [ output ] = applyRelationWeights( P, wO, wH, wMI, wNI, imageH, imageO )
%APPLYRELATIONWEIGHTS Summary of this function goes here
%   Detailed explanation goes here

oMI = logsigmoid(wMI*P);
oNI = logsigmoid(wNI*P);

[oM oMH] = applyWeights(oMI, imageH, imageO);
[oN oNH] = applyWeights(oNI, imageH, imageO);

% consolidate hidden layer input
oMNP = [oM ; oN ; P];

% get hidden layer output
oH = logsigmoid(wH*oMNP);

% get ouput layer output
oO = logsigmoid(wO*oH);

output = oO;