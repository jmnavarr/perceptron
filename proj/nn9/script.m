cap = 8;

if exist('imageH')
    
else
    [s s imageH imageO] = learnImages;
end

if exist('wO')
else
    [trainset testset wO wH imageO imageH errors] = learnRelations(imageH, imageO, cap);
end


% test all images
[images phrases] = getRelations(cap);
pbin = str2bin(phrases);
pbin = pbin(:,testset);

pim = applyRelationWeights(pbin, wO, wH, imageO, imageH);
phrases = phrases(testset);

[nim s]= size(phrases);

drawImages(pim,20,20,nim,phrases);

images = images(:,:,testset);
images = reshape(images,400,nim);
mse_testset = mean(mean((images - pim).^2))

% for i = 1:16:144
%     drawImages(pim(:,i:(i+15)),20,20,16,phrases(i:(i+15)));
% end

% add a new image, and see how it looks
[images phrases] = getRelations(cap+1);

checkset = (cap+1):(cap+1):(cap*(cap+1));
checkset = [checkset (cap*(cap+1)+1):(cap*(cap+1)+cap+1)];
checkset = [checkset (checkset + (cap*(cap+1)+cap+1)) (checkset + 2*(cap*(cap+1)+cap+1)) (checkset + 3*(cap*(cap+1)+cap+1))];

checksetslim = randsample(checkset, 16, false);

pbin = str2bin(phrases);
pbin = pbin(:,checksetslim);

pim = applyRelationWeights(pbin, wO, wH, imageO, imageH);
phrases = phrases(checksetslim);

[nim s]= size(phrases);

drawImages(pim,20,20,nim,phrases);

images = images(:,:,testset);
images = reshape(images,400,nim);
mse_extended = mean(mean((images - pim).^2))

figure;
plot(errors);
