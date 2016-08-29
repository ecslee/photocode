function [ RGB ] = LABImg2RGBImg(Img)

[ImgCol,ImgRow,ImgCh] = size(Img); 

for j=1:ImgRow
    for i = 1:ImgCol
        [r g b] = Lab2RGB(Img(i,j,1),Img(i,j,2),Img(i,j,3));  
        RGB(i,j,1) = r;  RGB(i,j,2) = g;  RGB(i,j,3) = b;
    end;
end;
