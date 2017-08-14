if exist('imageH')
    
else
    [s t imageH imageO] = learnImages;
end

if exist('wO')
else
    [input target wO wH imageO imageH] = learnRelations(imageH, imageO);
end


% test all images
[images phrases] = getRelations;
pbin = str2bin(phrases);
pbin = pbin(:,target);  % target is the output randomly selected testset returned from learnRelations

pim = applyRelationWeights(pbin, wO, wH, imageO, imageH);
phrases = phrases(target);

[nim nlen]= size(phrases);

drawImages(pim,20,20,nim,phrases);

% for i = 1:16:144
%     drawImages(pim(:,i:(i+15)),20,20,16,phrases(i:(i+15)));
% end
