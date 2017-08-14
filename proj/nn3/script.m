if(exist('imageH'))
    
else
    [s s imageH imageO] = learnImages;
end

[wO wH wMI wNI imageH imageO] = learnRelations(imageH, imageO);

