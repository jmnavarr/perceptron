function [ trainset, testset, wO, wH, imageO, imageH ] = learnRelations( imageH, imageO )
%LEARNRELATIONS Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
    [scratch scratch imageH imageO] = learnImages;
end

numHiddenNeurons = 100;
epochs = 1000;
goal_err = 10e-5;
drawrate = 25;
color = 'Gray';
testsetsize = 16;
imagedisplaysize = 16;
lrate = .005;

% get inputs
[images phrases] = getRelations;

% format the input character strings into binary strings
P = str2bin(phrases);

% image dimensions
[imRows, imCols, numIm] = size(images);
flatimlen = imRows*imCols;

% flatten target images into 1D vectors
T = reshape(images, flatimlen, numIm);


% restrict input
trainset = randsample(1:numIm,numIm-testsetsize,false);
P = P(:,trainset);
T = T(:,trainset);
phrases = phrases(trainset);

testset = setdiff(1:numIm,trainset);

sim = sort(randsample(1:floor(numIm-testsetsize),imagedisplaysize,false));
[eim nim] = size(sim);



% establish starting params
[phraselen numphrases] = size(P);
[modulehidden modinlen] = size(imageH);
[modoutlen scratch] = size(imageO);

a =  0.3;
b = -0.3;

% generate random weights
wH = a + (b-a)*rand(numHiddenNeurons,modoutlen*2 + phraselen - 2 * modinlen);
wO = a + (b-a)*rand(flatimlen,numHiddenNeurons);

% apply the image modules
[oM oMH] = applyWeights(P(1:modinlen,:), imageH, imageO);
[oN oNH] = applyWeights(P((phraselen-modinlen+1):phraselen,:), imageH, imageO);

% consolidate hidden layer input
oMNP = [oM ; P((modinlen+1):(phraselen-modinlen),:) ; oN ];

% get hidden layer output
oH = logsigmoid(wH*oMNP);

% get ouput layer output
oO = logsigmoid(wO*oH);

% calculate error
e = T - oO;
error = mean(mean(e.*e));

% display our current output
if drawrate > 0
    window = drawImages(oO(:,sim),imRows,imCols,nim,phrases(sim),color);
end
mseplot = 0;
epochplot = 0;

% do training
for  itr =1:epochs
    if error <= goal_err 
        break
    end

    % update weights

    % iterate across all input images
    for i = 1:numphrases
        % get derivatives at every layer
        dO = dlogsigmoid(oO(:,i));
        dH = dlogsigmoid(oH(:,i));
        
        % get error slopes for each layer
        eO = dO .* e(:,i);
        eH = dH .* (wO' * eO);

        % update weights
        wO = wO + lrate * eO * oH(:,i)';
        wH = wH + lrate * eH * oMNP(:,i)';

        % get new ouput
        % apply the image modules
        [oM(:,i) oMH(:,i)] = applyWeights(P(1:modinlen,i), imageH, imageO);
        [oN(:,i) oNH(:,i)] = applyWeights(P((phraselen-modinlen+1):phraselen,i), imageH, imageO);

        % consolidate hidden layer input
        oMNP(:,i) = [oM(:,i) ; P((modinlen+1):(phraselen-modinlen),i) ; oN(:,i) ];

        % get hidden layer output
        oH(:,i) = logsigmoid(wH*oMNP(:,i));

        % get ouput layer output
        oO(:,i) = logsigmoid(wO*oH(:,i));
    end

    % calculate new error
    e = T - oO;
    error = mean(mean(e.*e));

    disp(sprintf('Iteration :%5d        mse :%12.10f%',itr,error));
    mseplot(itr) = error;
    epochplot(itr) = itr;
    
    % every once in a while, visualise the outputs
    if mod(itr,drawrate) == 0 && drawrate > 0
        window = drawImages(oO(:,sim),imRows,imCols,nim,phrases(sim),color,window);
    end
end

% display the final output
if drawrate > 0
    window = drawImages(oO(:,sim),imRows,imCols,nim,phrases(sim),color,window);
end
