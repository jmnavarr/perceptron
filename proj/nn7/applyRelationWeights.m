function [ output ] = applyRelationWeights( P, wO, wH, imageO, imageH )
%APPLYRELATIONWEIGHTS Summary of this function goes here
%   Detailed explanation goes here

% get lengths
[phraselen numphrases] = size(P);
[modulehidden modinlen] = size(imageH);

% apply the image modules
[oM oMH] = applyWeights(P(1:modinlen,:), imageH, imageO);
[oN oNH] = applyWeights(P((phraselen-modinlen+1):phraselen,:), imageH, imageO);

% consolidate hidden layer input
oMNP = [oM ; P((modinlen+1):(phraselen-modinlen),:) ; oN ];

% get hidden layer output
oH = logsigmoid(wH*oMNP);

% get ouput layer output
oO = logsigmoid(wO*oH);

% assign output
output = oO;
