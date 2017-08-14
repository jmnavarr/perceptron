function [ win ] = testImages( window )
%TESTIMAGES Draw the input images
%   Detailed explanation goes here

if nargin < 1
    window = figure;
end

[images words] = getImages;

[x y z] = size(images);

drawImages(images, x, y, z, words, window);
win = window;
