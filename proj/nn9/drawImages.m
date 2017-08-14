function [ window ] = drawImages( images, rows, cols, numIm, titles, cspace, handle )
%DRAWIMAGES Summary of this function goes here
%   Detailed explanation goes here

if nargin < 6
    cspace = 'Gray';
end

if nargin < 7
    handle = figure;
else
    figure(handle);
end

plotWidth = ceil(sqrt(numIm));
plotHeight = ceil(numIm/plotWidth);

snapshot = reshape(images, rows, cols, numIm);
figure(handle);
for i=1:numIm
    im=snapshot(:,:,i);
    subplot(plotHeight,plotWidth,i);
    imshow(im);
    title(titles(i));
end
colormap(cspace);
drawnow;

window = handle;