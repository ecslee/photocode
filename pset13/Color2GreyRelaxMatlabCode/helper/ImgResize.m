function [ResizedImg]=ImgResize(InImg, RESIZE, imSize)
% function [InImgVec]=InImgXYZto3Vec(InImg)
% Input: w x h x component image
% returns w*h x component array (cols are components, rows are values)
% has option to RESIZE image with width = imSize

  [InImgCol,InImgRow,InImgCh] = size(InImg); 
    
if (RESIZE && InImgRow > imSize && InImgCol > imSize)
    if (0)
        Rsq = resize(double(InImg(:,:,1)), imSize);
        Gsq = resize(double(InImg(:,:,2)), imSize);
        Bsq = resize(double(InImg(:,:,3)), imSize);
    end;
    if (1)
        Rsq =  im_resize(double(InImg(:,:,1)), floor(imSize*InImgRow/InImgCol), floor(imSize*InImgCol/InImgCol));
        Gsq =  im_resize(double(InImg(:,:,2)), floor(imSize*InImgRow/InImgCol), floor(imSize*InImgCol/InImgCol));
        Bsq =  im_resize(double(InImg(:,:,3)), floor(imSize*InImgRow/InImgCol), floor(imSize*InImgCol/InImgCol));
    end;
    InImgRow = size(Rsq,2);
    InImgCol = size(Rsq,1);
else
    Rsq = InImg(:,:,1);
    Gsq = InImg(:,:,2);
    Bsq = InImg(:,:,3);
end;
ResizedImg=zeros(size(Rsq,1),size(Rsq,2),3);
ResizedImg(:,:,1)=Rsq;
ResizedImg(:,:,2)=Gsq;
ResizedImg(:,:,3)=Bsq;