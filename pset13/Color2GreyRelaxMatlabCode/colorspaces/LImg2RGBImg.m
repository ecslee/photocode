function [ RGB ] = LImg2RGBImg(Img)
%%Converts a single L (of LAB) channel image into 3 channel RGB image

[ImgCol,ImgRow,ImgCh] = size(Img); 

for j=1:ImgRow
    for i = 1:ImgCol
        [r g b] = Lab2RGB(Img(i,j,1),0,0);  
        RGB(i,j,1) = r;  RGB(i,j,2) = g;  RGB(i,j,3) = b;
    end;
end;
