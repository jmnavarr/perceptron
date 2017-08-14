[asciichar, asciicode] = textread('code.m','%s %s',-1);
reply = input('Enter text input word: ', 's');

replymat = zeros(inputX, inputY);
for i = 1:size(reply,2)
    for j = 1:size(asciichar,1)
        if(isequal(reply(i), char(asciichar(j))))
            replymat(i, 3:10) = char(asciicode(j)) - 48;
        end
    end
end
%replymat
%iword = 4; % index of input word

for iword = 1:numInput
    if(isequal(replymat, inword(:,:,iword)))
        break;
    else
        iword = -1;
    end
end

if (iword ~= -1)
    P = reshape(inword, inputX*inputY, numInput);
    for i = 1:Q
        A1(:,i)=logsigmoid(W1*P(:,i),b1); 
        A2(:,i)=logsigmoid(W2*A1(:,i),b2);
    end

    snapshot = reshape(real(A2), imX, imY, numIm);
    win = figure;
    set(gca,'Position',[0 0 1 1]);

    im=snapshot(:,:,iword);
    imshow(im);
end
