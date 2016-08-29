function [ImgVec]=ImgXYZto3Vec(InImg)
% function [ImgVec]=ImgXYZto3Vec(Img)
% Input: w x h x component image
% returns w*h x component array (cols are components, rows are values)
 
  [InImgCol,InImgRow,InImgCh] = size(InImg); 
     
%Want to represent a matrix of A such that row1 = R, row2 = G, row3=B
WH = InImgCol*InImgRow;
R =  reshape(InImg(:,:,1),1,WH);
G =  reshape(InImg(:,:,2),1,WH);
B =  reshape(InImg(:,:,3),1,WH);
RGBInImg = [R ;G; B];
 

%     figure(FigImages);
% subplot(gridw,gridh,(gridw*line)+2);
% Rimage = showGreyImage(R,InImgCol,InImgRow, MyXLabel);
% 
% subplot(gridw,gridh,(gridw*line)+3);
% Gimage = showGreyImage(G,InImgCol,InImgRow, MyYLabel); 
% 
% subplot(gridw,gridh,(gridw*line)+4);
% Bimage = showGreyImage(B,InImgCol,InImgRow, MyZLabel);

%PC_EVECTORS returns EigenVectors, EigenValues, and Psi=mean image
ImgVec = double(RGBInImg);

return;
