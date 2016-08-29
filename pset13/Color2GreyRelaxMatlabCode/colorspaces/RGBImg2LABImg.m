function [ LAB ] = RGBImg2LABImg(Img)

[ImgCol,ImgRow,ImgCh] = size(Img); 

for j=1:ImgRow
    for i = 1:ImgCol
        [L a b] = RGB2Lab(Img(i,j,1),Img(i,j,2),Img(i,j,3));  
        LAB(i,j,1) = L;  LAB(i,j,2) = a;  LAB(i,j,3) = b;
        if (MaxMax(L) > 100 || MinMin(L) < 0)
            fprintf('Error: L(%d,%d) = %f \n', i,j,L);
        end;
       if (MaxMax(a) > 500 || MinMin(a) < -500)
            fprintf('Error: a(%d,%d) = %f \n', i,j,L);
        end;
       if (MaxMax(b) > 500 || MinMin(b) < -500)
            fprintf('Error: b(%d,%d) = %f \n', i,j,L);
        end;
       
        
        %Normalized LAB
        %L = L/100.0*255.0;
        %a = (((a/120)+1)/2)*255.0;
        %b = (((b/120)+1)/2)*255.0;
        %clamp(L,0,255);clamp(a,0,255);clamp(b,0,255);
        %LAB_255(i,j,1) = L;  LAB_255(i,j,2) = a;  LAB_255(i,j,3) = b;
    end;
end;
