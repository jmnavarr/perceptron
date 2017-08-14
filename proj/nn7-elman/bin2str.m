function strOut = bin2str(binVector)
    binValues = [ 64 32 16 8 4 2 1 ];    
    binMatrix = reshape(binVector,7,[]);
    strOut = char(binValues*binMatrix);
end

