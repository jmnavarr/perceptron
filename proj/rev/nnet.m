% Neural net training for images

clear memory
clear all
clc

window = figure;
numHiddenNeurons = 10;
epochs = 10000;
goal_err = 10e-5;
drawrate = 50;
lrate=.1;

inputimages;
inputwords;

% image dimensions
[imX, imY, numIm] = size(images);

% target images
T = reshape(images, imX*imY, numIm);


% input data
[inputX, inputY, numInput] = size(inword);

P = reshape(inword, inputX*inputY, numInput);

plotWidth = ceil(sqrt(numIm));

% display the training images 
% figure;
% for i=1:numIm
%     im=images(:,:,i);
%     subplot(plotWidth,plotWidth,i),imshow(im);
% end

S1=numHiddenNeurons;   % number of hidden layers
S2=imX*imY;   % number of output layers (= number of classes)

[R,Q]=size(P);
a=0.3;                        % define the range of random variables
b=-0.3;
W1=a + (b-a) *rand(S1,R);     % Weights between Input and Hidden Neurons
W2=a + (b-a) *rand(S2,S1);    % Weights between Hidden and Output Neurons
b1=a + (b-a) *rand(S1,1);     % Weights between Input and Hidden Neurons
b2=a + (b-a) *rand(S2,1);     % Weights between Hidden and Output Neurons

n1=W1*P;
A1=logsigmoid(n1);
n2=W2*A1;
A2=logsigmoid(n2);

e=A2-T;
error =0.5* mean(mean(e.*e));    

%nntwarn off

for itr = 1:epochs
    if error <= goal_err 
        break
    else
         for i = 1:Q
            df1 = dlogsigmoid(n1(:,i),A1(:,i));
            df2 = dlogsigmoid(n2(:,i),A2(:,i));
            s2 = -2*diag(df2) * e(:,i);			       
            s1 = diag(df1)* W2'* s2;
            W2 = W2-lrate*s2*A1(:,i)';
            b2 = b2-lrate*s2;
            W1 = W1-lrate*s1*P(:,i)';
            b1 = b1-lrate*s1;

            A1(:,i) = logsigmoid(W1*P(:,i),b1);
            A2(:,i) = logsigmoid(W2*A1(:,i),b2);
         end
         e = T - A2;
         error = 0.5*mean(mean(e.*e));
         disp(sprintf('Iteration :%5d        mse :%12.6f%',itr,error));
         mse(itr)=error;
    end
    
    if mod(itr-1,drawrate) == 0
        snapshot = reshape(real(A2), imX, imY, numIm);
        figure(window);
        for i=1:numIm
            im=snapshot(:,:,i);
            subplot(plotWidth,plotWidth,i);
            imshow(im);
        end
        drawnow;
    end

end

threshold=0.9;   % threshold of the system (higher threshold = more accuracy)

% training images result

TrnOutput=reshape(real(A2), imX, imY, numIm);

% display the training images 

figure(window),
for i=1:numIm
    im=TrnOutput(:,:,i);
%    im=imresize(im,4,'nearest');        % resize the image to make it clear
    subplot(plotWidth,plotWidth,i),imshow(im);%title(strcat('Train image/Class #', int2str(ceil(i/n))))
end
drawnow;
