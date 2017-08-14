strLearnNewInput(1) = {'diagonal'};

imgLearnNewInput(:,:,1) =  [
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0;
    0	0	1	1	1	1	1	1	0	0;
    0	0	1	1	0	0	0	1	0	0;
    0	0	1	0	1	0	0	1	0	0;
    0	0	1	0	0	1	0	1	0	0;
    0	0	1	0	0  	0   1	1	0	0;
    0	0	1	1	1	1	1	1	0	0;
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0];

%addImageToLearn(strLearnNewInput, imgLearnNewInput);
addImageToLearnSingleNet(strLearnNewInput, imgLearnNewInput);