function [images, phrases] = getRelations( )
%GETRELATIONS get images and words of images placed relative to each other

relations(1) = {' left '};
relations(2) = {' right '};
relations(:,3) = {' above '};
relations(:,4) = {' below '};
% relations(:,5) = {' front '};
% relations(:,6) = {' behind '};

relstrings = char(relations)';

% get pics that we'll place
[pics, picnames] = getImages;

% get image size, so we know what the final image size
[picRows, picCols, npics] = size(pics);

imageRows = 2*picRows;
imageCols = 2*picCols;



% get lengths and what not of string inputs/outputs
[rlen nrels] = size(relstrings);
names = char(picnames)';
plen = size(names);

rellen = plen*2 + rlen;

% figure out relationship phrase assembly
seg1s = 1;
seg1e = plen;
seg2s = seg1e + 1;
seg2e = seg1e + rlen;
seg3s = seg2e + 1;
seg3e = rellen;

% preallocate for speed
nphrases = npics * npics * nrels;

images = zeros(imageRows,imageCols,nphrases);
phrases = cellstr(cast(zeros(nphrases), 'char'));


% curr image iterator
currIm = 1;

% assemble left

imrows = cast(picRows / 2, 'uint32') + 1;
imrowe = imrows + picRows - 1;

im1cols = 1;
im1cole = picCols;
im2cols = picCols+1;
im2cole = picCols*2;

for i = 1:npics
    for j = 1:npics
        % assemble phrase
        charphrase(seg1s:seg1e) = names(:,i);
        charphrase(seg2s:seg2e) = relstrings(:,1);
        charphrase(seg3s:seg3e) = names(:,j);
        
        phrases(currIm) = cellstr(charphrase);
        
        % assemble image
        images(imrows:imrowe,im1cols:im1cole,currIm) = pics(:,:,i);
        images(imrows:imrowe,im2cols:im2cole,currIm) = pics(:,:,j);

        % next im
        currIm = currIm+1;
    end
end


% assemble right

imrows = cast(picRows / 2, 'uint32') + 1;
imrowe = imrows + picRows - 1;

im2cols = 1;
im2cole = picCols;
im1cols = picCols+1;
im1cole = picCols*2;

for i = 1:npics
    for j = 1:npics
        % assemble phrase
        charphrase(seg1s:seg1e) = names(:,i);
        charphrase(seg2s:seg2e) = relstrings(:,2);
        charphrase(seg3s:seg3e) = names(:,j);
        
        phrases(currIm) = cellstr(charphrase);
        
        % assemble image
        images(imrows:imrowe,im1cols:im1cole,currIm) = pics(:,:,i);
        images(imrows:imrowe,im2cols:im2cole,currIm) = pics(:,:,j);

        % next im
        currIm = currIm+1;
    end
end


% assemble above

im1rows = 1;
im1rowe = picRows;
im2rows = picRows+1;
im2rowe = picRows*2;

imcols = cast(picCols / 2, 'uint32') + 1;
imcole = imcols + picCols - 1;

for i = 1:npics
    for j = 1:npics
        % assemble phrase
        charphrase(seg1s:seg1e) = names(:,i);
        charphrase(seg2s:seg2e) = relstrings(:,3);
        charphrase(seg3s:seg3e) = names(:,j);
        
        phrases(currIm) = cellstr(charphrase);
        
        % assemble image
        images(im1rows:im1rowe,imcols:imcole,currIm) = pics(:,:,i);
        images(im2rows:im2rowe,imcols:imcole,currIm) = pics(:,:,j);

        % next im
        currIm = currIm+1;
    end
end


% assemble below

im2rows = 1;
im2rowe = picRows;
im1rows = picRows+1;
im1rowe = picRows*2;

imcols = cast(picCols / 2, 'uint32') + 1;
imcole = imcols + picCols - 1;

for i = 1:npics
    for j = 1:npics
        % assemble phrase
        charphrase(seg1s:seg1e) = names(:,i);
        charphrase(seg2s:seg2e) = relstrings(:,4);
        charphrase(seg3s:seg3e) = names(:,j);
        
        phrases(currIm) = cellstr(charphrase);
        
        % assemble image
        images(im1rows:im1rowe,imcols:imcole,currIm) = pics(:,:,i);
        images(im2rows:im2rowe,imcols:imcole,currIm) = pics(:,:,j);

        % next im
        currIm = currIm+1;
    end
end
