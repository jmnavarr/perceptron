function [ input, target, wH, wO ] = learnImages(numHiddenNeurons, epochs, goal_err, lrate, drawrate, color)
% LEARNIMAGES Neural net training for images

if nargin < 1
    numHiddenNeurons = 20;
end

if nargin < 2
    epochs = 10000;
end

if nargin < 3
    goal_err = 10e-5;
end

if nargin < 4
    drawrate = 10000000;
end

if nargin < 5
    lrate=.2;
end

if nargin < 6
    %color = 'Jet';
    color = 'Gray';
end


% get input
[images, input] = getImages;

% format the input character strings into binary strings
P = str2bin(input);

% image dimensions
[imRows, imCols, numIm] = size(images);

% flatten target images into 1D vectors
T = reshape(images, imRows*imCols, numIm);

% establish starting params
[W1 W2] = createWeights(P, T, numHiddenNeurons);
[output hidden] = applyWeights(P, W1, W2);
e = T - output;
error = .5*mean(mean(e.*e));

% display our current output
window = drawImages(output,imRows,imCols,numIm,input,color);

% do training
for  itr =1:epochs
    if error <= goal_err 
        break
    end

    % update weights
    [W1, W2, error, output, hidden, e] = updateWeights(P, T, W1, W2, lrate, output, hidden, e);

    disp(sprintf('Iteration :%5d        mse :%12.6f%',itr,error));

    % every once in a while, visualise the outputs
    if mod(itr,drawrate) == 0
        drawImages(real(output),imRows,imCols,numIm,input,color,window);
    end
end

% display the final output
drawImages(output,imRows,imCols,numIm,input,color,window);

% assign output
input = P;
target = T;
wH = W1;
wO = W2;
