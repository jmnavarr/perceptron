iword = 1; % index of input word
for i = 1:Q
    A1(:,i)=logsig(W1*train(:,i,iword),b1); 
    A2(:,i)=logsig(W2*A1(:,i),b2);
end

%A2
TrnOutput = real(A2 > threshold)
