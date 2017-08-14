function [ bin ] = str2bin( strings )
%STR2BIN Convert an array of strings into an array of binary strings
%   Detailed explanation goes here

chars = char(strings)';

[len num] = size(chars);

bin = zeros(7*len,num);

for i = 1:num
    binmat = dec2bin(chars(:,i))' - '0';
    bin(:,i) = reshape(binmat, 7*len, 1);
end

