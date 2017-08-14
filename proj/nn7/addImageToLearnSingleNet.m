function [ images, input ] = addImageToLearnSingleNet(strInput, imgTarget)

% configure
numHiddenNeurons = 7;
epochs = 10000;
goal_err = 10e-12;
drawrate = 100;
lrate=.2;
color = 'Gray';

numPseudoPatterns = 15;

% get original weights
if exist('imageH') 
else
    [oInput oTarget imageH imageO] = learnImages(numHiddenNeurons, epochs, goal_err, lrate, drawrate, color);
end

[len num] = size(oInput);
[tLen tNum] = size(oTarget);

% generate input for pseudo-patterns
%pseudoInput = round(rand(len, numPseudoPatterns));
pseudoInput = randsample('abcdefghijklmnopqrstuvwxyz', (len/7)*numPseudoPatterns, true, []);
pseudoInput = reshape(str2bin(pseudoInput), len, numPseudoPatterns);

% get output for pseudo-patterns
[pO pH] = applyWeights(pseudoInput, imageH, imageO);

% test dimensions to make sure they match
P = str2bin(strInput);
[iStrLen iStrNum] = size(P);
if iStrLen ~= len
    error('Input string lengths do not match!');
end

[iImgRows, iImgCols, iNumImg] = size(imgTarget);
T = reshape(imgTarget, iImgRows*iImgCols, iNumImg);
[iImgLen, iImgNum] = size(T);
if iImgLen ~= tLen
    error('Input image dimensions do not match!');
end

% join pseudo-patterns and new input
pI = [pseudoInput P];
pO = [pO T];

% map strings to images
[len numin] = size(pI);
[imRows imCols scratch] = size(imgTarget);
for i = 1:numin
    strIn(i) = { bin2str(reshape(pI(:,i), 7, len/7)) };
    imgIn(:,:,i) = reshape(pO(:,i), imRows, imCols);
end

% display pseudo-patterns & input
[oO oH] = applyWeights(pI, imageH, imageO);
window = drawImages(oO,imRows,imCols,numin,strIn,color);

% train on pseudo-patterns & input
% [oO, wH, wO] = doBatchTraining(numHiddenNeurons, epochs, goal_err, lrate, drawrate, color, strIn, imgIn, pI, pO, imageH, imageO, window);
[oO, wH, wO] = doOnlineTraining(numHiddenNeurons, epochs, goal_err, lrate, drawrate, color, strIn, imgIn, pI, pO, imageH, imageO, window);


% display learned input & pseudo-patterns
drawImages(oO,imRows,imCols,numin,strIn,color,window);



% % do transfer phase
% [len num] = size(oInput);
% pseudoInput = randsample('abcdefghijklmnopqrstuvwxyz', (len/7)*numPseudoPatterns, true, []);
% pseudoInput = reshape(str2bin(pseudoInput), len, numPseudoPatterns);
% 
% % get output for pseudo-patterns
% [pO pH] = applyWeights(pseudoInput, wH, wO);
% 
% [len numin] = size(pseudoInput);
% [imRows imCols scratch] = size(imgTarget);
% for i = 1:numin
%     strIn(i) = { bin2str(reshape(pseudoInput(:,i), 7, len/7)) };
%     imgIn(:,:,i) = reshape(pO(:,i), imRows, imCols);
% end
% 
% window = drawImages(pO,imRows,imCols,numin,strIn,color);
% [oO, wH, wO] = doOnlineTraining(numHiddenNeurons, 10, goal_err, lrate, drawrate, color, strIn, imgIn, pI, pO, imageH, imageO, window);


% use new weights to display original images - hopefully without catastrophic forgetting!
pI = [oInput P];    % pO = [oTarget T];

% get strings
[len numin] = size(pI);
for i = 1:numin
    strIn(i) = { bin2str(reshape(pI(:,i), 7, len/7)) };
end

[oO oH] = applyWeights(pI, wH, wO);
drawImages(oO,imRows,imCols,numin,strIn,color);

