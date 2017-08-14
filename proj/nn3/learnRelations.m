function [ wO, wH, wMI, wNI, imageH, imageO ] = learnRelations( imageH, imageO )
%LEARNRELATIONS Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
    [scratch scratch imageH imageO] = learnImages;
end

numHiddenNeurons = 100;
epochs = 10000;
goal_err = 10e-5;
drawrate = 1;
lrate=.001;
color = 'Gray';


sim = 1:20:144;
[eim nim] = size(sim);


% get inputs
[images phrases] = getRelations;

% restrict input
% images = images(:,:,28:36);
% phrases = phrases(28:36);

% format the input character strings into binary strings
P = str2bin(phrases);

% image dimensions
[imRows, imCols, numIm] = size(images);
flatimlen = imRows*imCols;

% flatten target images into 1D vectors
T = reshape(images, flatimlen, numIm);

% establish starting params
[phraselen numphrases] = size(P);
[modulehidden modinlen] = size(imageH);
[modoutlen scratch] = size(imageO);

a =  0.3;
b = -0.3;

% generate random weights
wMI = a + (b-a)*rand(modinlen,phraselen);
wNI = a + (b-a)*rand(modinlen,phraselen);
wH = a + (b-a)*rand(numHiddenNeurons,modoutlen*2 + phraselen);
wO = a + (b-a)*rand(flatimlen,numHiddenNeurons);

% apply the image modules
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

% calculate error
e = T - oO;
error = mean(mean(e.*e));

% display our current output
if drawrate > 0
    window = drawImages(oO(:,sim),imRows,imCols,nim,phrases(sim),color);
end

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
        dM = dlogsigmoid(oM(:,i));
        dN = dlogsigmoid(oN(:,i));
        dMH = dlogsigmoid(oMH(:,i));
        dNH = dlogsigmoid(oNH(:,i));
        dMI = dlogsigmoid(oMI(:,i));
        dNI = dlogsigmoid(oNI(:,i));
        
        % get error slopes for each layer
        eO = dO .* e(:,i);
        eH = dH .* (wO' * eO);
        eMNP = wH' * eH;
        eM = dM .* eMNP(1:modoutlen);
        eN = dN .* eMNP((modoutlen+1):(2*modoutlen));
        eMH = dMH .* (imageO' * eM);
        eNH = dNH .* (imageO' * eN);
        eMI = dMI .* (imageH' * eMH);
        eNI = dNI .* (imageH' * eNH);

        % update weights
        wO = wO + lrate * eO * oH(:,i)';
        wH = wH + lrate * eH * oMNP(:,i)';
        wMI = wMI + lrate * eMI * P(:,i)';
        wNI = wNI + lrate * eNI * P(:,i)';

        % get new output
        % apply the image modules
        oMI(:,i) = logsigmoid(wMI*P(:,i));
        oNI(:,i) = logsigmoid(wNI*P(:,i));

        [oM(:,i) oMH(:,i)] = applyWeights(oMI(:,i), imageH, imageO);
        [oN(:,i) oNH(:,i)] = applyWeights(oNI(:,i), imageH, imageO);

        % consolidate hidden layer input
        oMNP(:,i) = [oM(:,i) ; oN(:,i) ; P(:,i)];

        % get hidden layer output
        oH(:,i) = logsigmoid(wH*oMNP(:,i));

        % get ouput layer output
        oO(:,i) = logsigmoid(wO*oH(:,i));
    end

    % calculate new error
    e = T - oO;
    error = mean(mean(e.*e));

    disp(sprintf('Iteration :%5d        mse :%12.6f%',itr,error));

    % every once in a while, visualise the outputs
    if mod(itr,drawrate) == 0 && drawrate > 0
        window = drawImages(oO(:,sim),imRows,imCols,nim,phrases(sim),color,window);
    end
end

% display the final output
if drawrate > 0
    window = drawImages(oO(:,sim),imRows,imCols,nim,phrases(sim),color,window);
end
 
