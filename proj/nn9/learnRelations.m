function [ trainset, testset, wO, wH, imageO, imageH, errors ] = learnRelations( imageH, imageO, cap )
%LEARNRELATIONS Summary of this function goes here
%   Detailed explanation goes here

% get image module weights, if we weren't given them
if nargin < 2
    [scratch scratch imageH imageO] = learnImages;
end

% params
numHiddenNeurons = 50;
epochs = 1000;
goal_err = 10e-7;
drawrate = 25;
color = 'Gray';
testsetsize = 16;
imagedisplaysize = 16;

errors = zeros(epochs,1);

% power of our modified error function
pow = 6;

% get inputs
[images phrases] = getRelations(cap);


% format the input character strings into binary strings
P = str2bin(phrases);

% image dimensions
[imRows, imCols, numIm] = size(images);
flatimlen = imRows*imCols;

% flatten target images into 1D vectors
T = reshape(images, flatimlen, numIm);


% restrict input
trainset = randsample(1:numIm,numIm-testsetsize,false);
P = P(:,trainset);
T = T(:,trainset);
phrases = phrases(trainset);

testset = setdiff(1:numIm,trainset);

sim = sort(randsample(1:floor(numIm-testsetsize),imagedisplaysize,false));
[eim nim] = size(sim);


% establish starting params
[phraselen numphrases] = size(P);
[modulehidden modinlen] = size(imageH);
[modoutlen scratch] = size(imageO);

a =  0.05;
b = -0.05;

% generate random weights
%numHiddenNeurons = modoutlen*2 + phraselen - 2 * modinlen;
wH = a + (b-a)*rand(numHiddenNeurons,modoutlen*2 + phraselen - 2 * modinlen);
wO = a + (b-a)*rand(flatimlen,numHiddenNeurons);

wHf = wH;
wOf = wO;

wHl = wH;
wOl = wO;

% apply the image modules
[oM oMH] = applyWeights(P(1:modinlen,:), imageH, imageO);
[oN oNH] = applyWeights(P((phraselen-modinlen+1):phraselen,:), imageH, imageO);

% consolidate hidden layer input
oMNP = [oM ; P((modinlen+1):(phraselen-modinlen),:) ; oN ];

% get hidden layer output
oH = logsigmoid(wH*oMNP);

% get ouput layer output
oO = logsigmoid(wO*oH);

% calculate error
e = T - oO;
error = mean(mean(e.*e));
e = e.*abs(e).^(pow-2);
lasterror = error;

% display our current output
if drawrate > 0
    window = drawImages(oO(:,sim),imRows,imCols,nim,phrases(sim),color);
end

% initial params
etaplus = 1.2;
etaminus = .5;
maxstep = 50;
minstep = 0;
baselrate = .0001;       % initial learn rate, should be pretty insensitive
dEwO = zeros(size(wO)); % last step's derivative of the error in the Output layer wrt output weights
dEwH = zeros(size(wH)); % last step's derivative of the error in the Hidden layer wrt hidden weights

deltaEwO = ones(size(wO))*baselrate; % the step deltas for the output weights
deltaEwH = ones(size(wH))*baselrate; % the step deltas for the hidden weights


wHbest = wH;
wObest = wO;
errorbest = error;


iterset = randsample(1:numphrases,numphrases);

% do training
for  itr =1:epochs
    if error <= goal_err 
        break
    end

    % current step's derivates of the errors wrt to weights
    wOb = zeros(size(wO));
    wHb = zeros(size(wH));
    
   %iterset = randsample(1:numphrases,floor(numphrases/16));

    % iterate across all input images
    for i = iterset
        % get derivatives of layer outputs
        dO = dlogsigmoid(oO(:,i));
        dH = dlogsigmoid(oH(:,i));

        % get error slopes for each layer
        eO = dO .* e(:,i);
        eH = dH .* (wO' * eO);

        % batch the weight changes
        wOb = wOb - eO * oH(:,i)';
        wHb = wHb - eH * oMNP(:,i)';
    end
    
    % determine weight changes in output layer
    [maxi maxj] = size(wO);
    signmap = sign(dEwO .* wOb);
    for i = 1:maxi
        for j = 1:maxj
            % determine direction of change
            if signmap(i,j) > 0
                % if sign agrees, push up the learn rate
                deltaEwO(i,j) = min(etaplus*deltaEwO(i,j),maxstep);
                
                % apply weight change
                wOf(i,j) = wO(i,j) - sign(wOb(i,j))*deltaEwO(i,j);
                
            elseif signmap(i,j) < 0
                oldWS = deltaEwO(i,j);
                
                % if signs don't agree, decrease step
                deltaEwO(i,j) = max(etaminus*deltaEwO(i,j),minstep);
                
                % roll back weight only if overall error increased
                if error > lasterror
                    wOf(i,j) = wO(i,j) + sign(dEwO(i,j))*oldWS;
                end
                
                % prevent weight step change next iteration
                wOb(i,j) = 0;

            else
                % sign is 0
                
                % apply weight change without change weight step
                wOf(i,j) = wO(i,j) - sign(wOb(i,j))*deltaEwO(i,j);
                
            end
        end
    end

    
    % store old weights
    wOl = wO;
    wO = wOf;
    
%     % adjust weights
%     deltaw = sign(wOb) .* deltaEwO;
%     wO = wO - deltaw;

    dEwO = wOb;
    
    
    
    
    % determine weight changes in hidden layer
    [maxi maxj] = size(wH);
    signmap = sign(dEwH .* wHb);
    for i = 1:maxi
        for j = 1:maxj
            % determine direction of change
            if signmap(i,j) > 0
                % if sign agrees, push up the learn rate
                deltaEwH(i,j) = min(etaplus*deltaEwH(i,j),maxstep);
                
                % apply weight change
                wHf(i,j) = wH(i,j) - sign(wHb(i,j))*deltaEwH(i,j);

            elseif signmap(i,j) < 0
                oldWS = deltaEwH(i,j);
                
                % if signs don't agree, decrease step
                deltaEwH(i,j) = max(etaminus*deltaEwH(i,j),minstep);
                
                % rollback weight only if error increased
                if error > lasterror
                    wHf(i,j) = wH(i,j) + sign(dEwH(i,j))*oldWS;
                end
                
                % prevent weight step change next iteration
                wHb(i,j) = 0;
            else
                % sign is 0
                
                % apply weight change without adjusting weightstep
                wHf(i,j) = wH(i,j) - sign(wHb(i,j))*deltaEwH(i,j);
            end
        end
    end

    % store old weights
    wHl = wH;
    wH = wHf;

%     % adjust weights
%     deltaw = sign(wHb) .* deltaEwH;
%     wH = wH - deltaw;

    dEwH = wHb;

%     wO = wO - .0004 * wOb;
%     wH = wH - .0004 * wHb;


    % get new outputs
    % apply the image modules
    [oM oMH] = applyWeights(P(1:modinlen,:), imageH, imageO);
    [oN oNH] = applyWeights(P((phraselen-modinlen+1):phraselen,:), imageH, imageO);

    % consolidate hidden layer input
    oMNP = [oM ; P((modinlen+1):(phraselen-modinlen),:) ; oN ];

    % get hidden layer output
    oH = logsigmoid(wH*oMNP);

    % get ouput layer output
    oO = logsigmoid(wO*oH);

    
    % calculate new error
    lasterror = error;
    e = T - oO;
    error = mean(mean(e.*e));
    e = e.*abs(e).^(pow-2);

    errors(itr) = error;
    
    if error < errorbest
        wHbest = wH;
        wObest = wO;
        besterror = error;
    end

    disp(sprintf('Iteration :%5d        mse :%12.10f%',itr,error));

    % every once in a while, visualise the outputs
    if mod(itr,drawrate) == 0 && drawrate > 0
        window = drawImages(oO(:,sim),imRows,imCols,nim,phrases(sim),color,window);
    end
end

% display the final output
if drawrate > 0
    window = drawImages(oO(:,sim),imRows,imCols,nim,phrases(sim),color,window);
end

wH = wHbest;
wO = wObest;