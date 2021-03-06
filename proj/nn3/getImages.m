function [images, input] = getImages( )
%GETIMAGES Use this to get the input images and words

input(1) = {'ghost'};

images(:,:,1) =  [
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0];


input(2) = {'blizzard'};

images(:,:,2) =  [
    1	1	1	1	1	1	1	1	1	1;
    1	1	1	1	1	1	1	1	1	1;
    1	1	1	1	1	1	1	1	1	1;
    1	1	1	1	1	1	1	1	1	1;
    1	1	1	1	1	1	1	1	1	1;
    1	1	1	1	1	1	1	1	1	1;
    1	1	1	1	1	1	1	1	1	1;
    1	1	1	1	1	1	1	1	1	1;
    1	1	1	1	1	1	1	1	1	1;
    1	1	1	1	1	1	1	1	1	1];



input(3) = {'tree'};

images(:,:,3) = [
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	1	0	0	0	0	0;
    0	0	0	1	1	1	0	0	0	0;
    0	0	0	1	1	1	0	0	0	0;
    0	0	0	0	1	0	0	0	0	0;
    0	0	0	0	1	0	0	0	0	0;
    0	0	0	0	1	0	0	0	0	0;
    0	0	0	1	1	1	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0];



input(4) = {'house'};

images(:,:,4) =  [
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	1	0	0	1	0	0;
    0	0	0	1	1	1	0	1	0	0;
    0	0	1	1	1	1	1	1	0	0;
    0	1	1	1	1	1	1	1	1	0;
    0	1	0	1	1	0	0	1	1	0;
    0	1	0	1	1	0	0	1	1	0;
    0	1	1	1	1	1	1	1	1	0;
    0	1	1	1	1	1	1	1	1	0;
    0	0	0	0	0	0	0	0	0	0];



input(5) = {'car'};

images(:,:,5) =  [
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0;
    0	0	1	1	1	1	0	0	0	0;
    0	1	0	1	0	0	1	0	0	0;
    0	1	0	1	0	0	0	1	0	0;
    0	1	1	1	1	1	1	1	1	0;
    0	1	1	1	1	1	1	1	1	0;
    0	0	1	1	0	0	1	1	0	0;
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0];



input(:,6) = {'window'};

images(:,:,6) =  [
    0	0	0	0	0	0	0	0	0	0;
    0	1	1	1	1	1	1	1	0	0;
    0	1	0	0	1	0	0	1	0	0;
    0	1	0	0	1	0	0	1	0	0;
    0	1	1	1	1	1	1	1	0	0;
    0	1	0	0	1	0	0	1	0	0;
    0	1	0	0	1	0	0	1	0	0;
    0	1	1	1	1	1	1	1	0	0;
    0	0	0	0	0	0	0	0	0	0;
    0	0	0	0	0	0	0	0	0	0];
