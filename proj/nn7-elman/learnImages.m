function [ input, target, wH, wO, errors ] = learnImages(numHiddenNeurons, epochs, goal_err, lrate, drawrate, color)
% LEARNIMAGES Neural net training for images

if nargin < 1
    numHiddenNeurons = 7;
end

if nargin < 2
    epochs = 10000;
end

if nargin < 3
    goal_err = 10e-12;
end

if nargin < 4
    drawrate = 100;
end

if nargin < 5
    lrate=.2;
end

if nargin < 6
    %color = 'Jet';
    color = 'Gray';
end

if nargin < 7
    batch = 1; % batch learning by default
end

% get input
[images, input] = getImages;

% format the input character strings into binary strings
P = str2bin(input);

% image dimensions
[imRows, imCols, numIm] = size(images);

% flatten target images into 1D vectors
T = reshape(images, imRows*imCols, numIm);

% establish starting params
[wH wO] = createWeights(P, T, numHiddenNeurons);

% display our current output
[oO oH] = applyWeights(P, wH, wO);
e = T - oO;
error = mean(mean(e.*e));
window = drawImages(oO,imRows,imCols,numIm,input,color);

% do training
%[oO, wH, wO] = doBatchTraining(numHiddenNeurons, epochs, goal_err, lrate, drawrate, color, input, images, P, T, wH, wO, window);
[oO, wH, wO] = doOnlineTraining(numHiddenNeurons, epochs, goal_err, lrate, drawrate, color, input, images, P, T, wH, wO, window);

% begin Elman
% I = P;
% O = T;
% Oseq = con2seq(O);
% I = [zeros(size(O,1) - size(I,1), 8); I];
% Iseq = con2seq(I);
% net = newelm([0 1],[7 1],{'tansig','logsig'});
% net.trainParam.epochs = 1000;
% net.trainParam.min_grad = 0;
% net.trainParam.lr = 0.2;
% net = train(net,Iseq{1,1}',Oseq{1,1}'); % this only learns ghost. how to make it learn all?
%                                         % error with other images does not
%                                         % go below 0.01
% for i = 1:1:8
%     oO(:,i) = sim(net, Iseq{1,i}')';
% end
% end Elman

% display the final output
drawImages(oO,imRows,imCols,numIm,input,color,window);

% assign output
input = P;
target = T;