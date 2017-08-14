clear memory
clear all
clc

lrate = 0.2;        % learning rate to find change in weight
threshold = 0.7;    % threshold of the system (higher threshold = more accuracy)
epochs = 1000;      % number of iterations
goal_err = 10e-5;    % goal error
error = 1;          % initial error
a = 0.3;                        % define the range of random variables
b = -0.3;
            
trainimg;
inwords;
[R,Q,Z]=size(inword);
S1 = 5;             % number of hidden layers
S2 = Q; %10;             % number of output layers (= number of classes)
nntwarn off

iword = 1;
W1 = a + (b-a) * rand(S1,R);     % Weights between Input and Hidden Neurons
W2 = a + (b-a) * rand(S2,S1);    % Weights between Hidden and Output Neurons
b1 = a + (b-a) * rand(S1,1);     % Weights between Input and Hidden Neurons
b2 = a + (b-a) * rand(S2,1);     % Weights between Hidden and Output Neurons

% multiply weights with layers
n1 = W1*train(:,:,iword);    % (weight between input and hidden) * (training input)
A1 = logsig(n1);  % logsig of (weight between input and hidden) * (training input)

n2 = W2*A1;   % (weight between hidden and output) * (logsig of input result)
A2 = logsig(n2);  % logsig of 

% calculate error
e = A2 - train(:,:,iword);
error = 0.5* mean(mean(e.*e));   
            
for  itr = 1:epochs
    if error <= goal_err 
        break
    else
        for iword = 1:Z                     
            for i = 1:Q
                df1 = dlogsig(n1,A1(:,i));
                df2 = dlogsig(n2,A2(:,i));
                s2 = -2 * diag(df2) * e(:,i);			       
                s1 = diag(df1) * W2' * s2;

                W2 = W2-lrate*s2*A1(:,i)'; % s2 is partial derivative, A1 is output of previous neuron
                b2 = b2-lrate*s2;
                W1 = W1-lrate*s1*train(:,i,iword)'; %s is partial derivative, P is output of previous neuron
                b1 = b1-lrate*s1;

                % calculate new output based on new weight
                A1(:,i)=logsig(W1*train(:,i,iword),b1); 
                A2(:,i)=logsig(W2*A1(:,i),b2);
             end
                e = train(:,:,iword) - A2;
                error = 0.5*mean(mean(e.*e));
                disp(sprintf('Iteration :%5d        mse :%12.6f%',itr,error));
                mse(itr)=error;
        end
    end
end

% training images result
% TrnOutput=real(A2)
% imshow(A2)
TrnOutput = real(A2 > threshold)    