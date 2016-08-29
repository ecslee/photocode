function [] = Color2Grey(fileName)

ImgPath = './';

%% Color2Grey via Relaxation  
%%  Copyright 2005  

%% Basic idea is to take the changes in Color image (Cij & Tij) and make a
%% new Gray image where the difference between dGij and Tij are minimized

%Since this is matlab, it is really easy to do this as a vector
% operation... the problem is that this won't work for large
% image, but this matlab code is just for prototyping

%%%%%%%%%%%%%%%%%%%%%%%%%  INITIALIZE VARIABLES   %%%%%%%%%%%%%%%%%%%%%%%%%

% Since the difference in A & B channels are large, we scale them  using
% sigmoid like function tanh
crunchTop = 8;

%This code has a default theta of 135 degrees, using 
%cool =  Blue/Green  warm = yellow/red

%Number of iterations for relaxing image
MAX_ITER = 15;
MIN_ITER = 10;

%flag which allows us to stop early if it converges before MAX_ITER
STOP_CONSTRAINT = 0; %Init flag  %Allows it to stop early by setting this flag to 1 

% Add option for adding random offset if mean difference is 0
USE_RANDOM_BUMP = 0;  %In practice, this just makes it oscilate longer...

%Pick little increment to add or subtract from G to make it get close to
%color image
little = 1; %.5;

USE_OPTIMIZED_T = 1; %Take advantage of T as 0 on diag, upper/lower symmetric
T_WH = 0;  %size of vector T when using vector rep for symmetric upper/lower triangle

%For plotting figures
gridw = 5; gridh = 1; line = 1;
DISPLAY = 1;

%If the image is too large, it takes too long or may run out of memory
% If RESIZE == 1, then we resize image to have width of imSize and
% proportional height
RESIZE = 1; imSize = 10;    



%%%%%%%%%%%%%%%%%%%%  LOAD IMAGES   %%%%%%%%%%%%%%%%%%%%%%%%%

ImgO = imread(strcat(ImgPath, fileName));
figHandle = figure ;
if (RESIZE)
    Img = ImgResize(ImgO,RESIZE,imSize); Img = double(Img);
else  
    Img = double(ImgO); imSize = size(Img,2);
end;
Img=clampM(Img,0,255.0); 

% [Img fileName figHandle] = myloadResizedImage(imgNum,gridw,gridh,RESIZE,imSize);
fprintf('Starting Relaxation on image: %s\n', fileName);


[ImgCol,ImgRow,ImgCh] = size(Img); 

%%%%%%%%%%%%%%%%%%%%  CONVERT IMAGE TO LAB   %%%%%%%%%%%%%%%%%%%%%%%%%

% Convert Image to Luminance Image:
fprintf('\tConvert image to LAB....\n');
LABImg = RGBImg2LABImg(Img);

NewLum = LABImg(:,:,1);

%Save a version of the default luminance
L2RGBImg = LImg2RGBImg(NewLum);

%%%%%%%%%%%%%%%%%%%%  Convert Image to VECTOR  %%%%%%%%%%%%%%%%%%%%%%%%%
%% it is faster in matlab to deal with images as vector.. 
%for loops are expensive

[LABImgVec] = ImgXYZto3Vec(LABImg);  %An array of (components) x (w*h) 
[Comp WH] = size(LABImgVec);

%%%%%%%%%%%%%%%% Variable INIT: string for saving files  %%%%%%%%%%%%%%%%%%%%%%%%%

filePATH = './';
str = sprintf('%sResArray_BGCool_crunchDiff%d_little%.2f_%s',filePATH,crunchTop,little,fileName);
str_grayImg  = sprintf('%sNewG%d_BGCool_crunchDiff%d_little%.2f_%s',filePATH,imSize,crunchTop,little,fileName);


%%%%%%%%%%%%%%%% PRECOMPUTE:  Color & Luminance differences %%%%%%%%%%%%%%%%%%%

fprintf('\tPrecompute Chrominance & Luminance differences....\n');
%Load file if can:
try
    filePATH = './';
    fmat = sprintf('%s%s_size%d.mat',filePATH,fileName,imSize);
    load(fmat, 'T','C','dL');
    printf('\t\tLoaded matrices from file\n');
catch
    fprintf('\t Was NOT able to find mat file: %s\n Precomputing...\n',fmat)
    
    % Algorithm:
    
    tic
    T = zeros(WH,WH);
    for i=1:WH
        for j = 1:WH
            
            if (i == j)
                T(i,j)=0;
            elseif (i > j) 
                T(i,j) = CalcTij(i,j, LABImgVec,crunchTop);
                T(j,i) = T(i,j)*-1;
            end;
        end; %for i creating C,T,dL
    end; %for j creating C,T,dL
    toc
    
    
    % If we didn't load up this mat  
    % Save out what we have so far
    filePATH = './';
    fmat = sprintf('%s%s_size%d.mat',filePATH,fileName,imSize);
    save(fmat);
    
end;% end of try & catch to read in matrices


%%%%%%%%%%%%%%%% Init Gray Image  %%%%%%%%%%%%%%%%%%%%%%%%%

% Init Gray image to 0
% G =  zeros(1,WH);
% OR Init Gray image to Luminance Image
G = LABImgVec(1,:);
updatedG = LABImgVec(1,:); %double buffer variable
maxL = MaxMax(NewLum);
minL = MinMin(NewLum);

% Set Neighborhood to consider to be N or something smaller
% If wanted to do something smaller, would have to use square matrix for G
% or do some very smart accounting
N = WH;
M = N;



%%%%%%%%%%%%%%%% Start Iterations  %%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\tStart Iterations on image....\n');

iter_want = zeros(MAX_ITER,1);
Tvector=zeros(1,WH);

tic
for iter = 1 : MAX_ITER 
    tic
    min_want = zeros(N,1);
    
    
    
    % Touch every pixel once, changing it a little
    % For now assume want to encode signed differences
    if (STOP_CONSTRAINT == 0)
        
        %%%%%%%%%%%%%%%% Main Loop  %%%%%%%%%%%%%%%%%%%%%%%%%
        for i=1:N
            
            
            Tvector(:)=T(i,:);
            
            
            % dG  = G(i) - G;
            %  dGPlus   = dG + (little*dG);
            %  dGNeg = dG - (little*dG);
            %  dGZero = dG;
            dGZero = G(i) - G;
            dGPlus = dGZero  +little; 
            dGNeg = dGZero-little ;
            
            %More continuous changes
            tmpD = (dGPlus - Tvector);
            want0_p = mean(tmpD.*tmpD);
            tmpD = (dGZero - Tvector);
            want0_0 = mean(tmpD.*tmpD);
            tmpD = (dGNeg - Tvector);
            want0_n = mean(tmpD .* tmpD);
            
            
            %if I add a little to Gi and it make dG - T(i,j) smaller, 
            %   this is good
            % otherwise if I subtract a little to Gi and it makes dG - T(i,j)
            % smaller, 
            %    do that
            %  else   leave it alone
            min_want(i) = min(min(want0_p,want0_0),want0_n);
            if (USE_RANDOM_BUMP && min_want(i) == want0_0 && min_want(i) == 0 )
                
                updatedG(i) = G(i)+ (rand(1)-.5); %Add random amount [-.5,.5]
                
                %fprintf(' adding random shift G(%d)\n',i);
                %Do nothing
            else
                if (min_want(i) == want0_p)
                    % fprintf('adding a little G(%d)\n',i);
                    updatedG(i) = G(i)+little;
                    %  fprintf(' adding  G(%d)\n',i);
                elseif (min_want(i) == want0_n)
                    % fprintf('subtracting a little G(%d)\n',i);
                    updatedG(i) = G(i) - little;
                    %  fprintf(' subtracting G(%d)\n',i);
                else 
                    if (USE_RANDOM_BUMP && mean((dGPlus - T(i,:)) .* (dGNeg - T(i,:))) >  (mean((dGPlus - T(i,:))))^2)
                        
                        updatedG(i) = G(i)+ (rand(1)-.5); 
                        
                        %    fprintf(' adding random shift G(%d)\n',i);
                        %else
                        %    fprintf('ick: doing nothing for G(%d)\n',i);
                    end;
                    %Do nothing
                end;
            end;
            
        end; %end of i, touching every pixel once & moving G[i] a little 
        toc
        %%%%%%%%%%%%%%%% END OF Main Loop  %%%%%%%%%%%%%%%%%%%%%%%%%
        
        G =updatedG;  %Double buffering
        
        if (   DISPLAY)
            %%%%%%%%%%%   DISPLAY Intermediate Results  %%%%%%%%%%%%%
            %Show results
            NewG = reshape(G,ImgCol,ImgRow);
            NewG2RGBImg = LImg2RGBImg(NewG);
            %imwrite(uint8(RGBImg), colormap(gray(256)), char(strcat('NewGrayRegularization_', fileName)), 'png');
            
            
            %line = 2; bin = 4;
            
            gridw = 3;
            gridh = 1;
            line = 1;
            
            subplot(gridw,gridh,line); 
            
            %Show Image
            image(NewG2RGBImg/255.0)
            title 'NewGray';
            %colormap(gray(256));
            daspect([1 1 1]);
            axis off;
            line = line + 1;
            
            subplot(gridw,gridh,line); 
            %Show Image
            image(Img/255.0)
            title 'Original';
            %colormap(gray(256));
            daspect([1 1 1]);
            axis off;
            line = line + 1;
            
            subplot(gridw,gridh,line)  %m x n matrix of figures
            %Show Image
            image(L2RGBImg/255.0)
            title 'Old L in RGB';
            colormap(gray(256));
            daspect([1 1 1]);
            axis off;
            line = line +1;
            
            iter_want(iter) =   mean(min_want);
            fprintf('\t%d: Mean min_want = %f\n', iter, mean(min_want));
            
            
            pause(.25);
            
            %%%%%%%%%%%   END OF  DISPLAY Intermediate Results  %%%%%%%%%%%%%
        end; %end of if DISPLAY
        
        if (iter > MIN_ITER)
            %%%%%%%%%%%   Other Stopping Criteria  %%%%%%%%%%%%%
            %                 if ( iter_want(iter-1) == iter_want(iter))
            %                     %Wait for user to say continue
            %                     mean(iter_want(iter-5:iter))
            %                     iter_want(iter) 
            %                     fprintf('----> Press any key to continue\n');
            %                     pause 
            %                 end;
            
            %%%%%%%%%%%   Other Stopping Criteria  %%%%%%%%%%%%%
            if (abs(mean(iter_want(iter-5:iter)) - iter_want(iter) ) < .001 ) 
                fprintf('\t stopping.. iterations no changing\n');
                STOP_CONSTRAINT = 1;
            end;
        end;
        
        %%%%%%%%%%%   Save out Intermediate Results  %%%%%%%%%%%%%
        NewG = reshape(G,ImgCol,ImgRow);
        NewG2RGBImg = LImg2RGBImg(NewG);
        fprintf('Will try to write out image %s\n', str_grayImg);
        imwrite(uint8(NewG2RGBImg), colormap(gray(256)), str_grayImg, 'png');
        
    end; %end of if stop constraint
end; %end of iter



%%%%%%%%%%%   END OF ALL for this image  %%%%%%%%%%%%%
NewG = reshape(G,ImgCol,ImgRow);


NewG2RGBImg = LImg2RGBImg(NewG);
%imwrite(uint8(RGBImg), colormap(gray(256)), char(strcat('NewGrayRegularization_', fileName)), 'png');


%line = 2; bin = 4;

gridw = 3;
gridh = 1;
line = 1;

subplot(gridw,gridh,line); 

%Show Image
image(NewG2RGBImg/255.0)
title 'NewGray';
%colormap(gray(256));
daspect([1 1 1]);
axis off;
line = line + 1;

subplot(gridw,gridh,line); 
%Show Image
image(Img/255.0)
title 'Original';
%colormap(gray(256));
daspect([1 1 1]);
axis off;
line = line + 1;

subplot(gridw,gridh,line)  %m x n matrix of figures
%Show Image
image(L2RGBImg/255.0)
title 'Old L in RGB';
colormap(gray(256));
daspect([1 1 1]);
axis off;

pause(2);


fprintf('Outputing Image array to file\n\t %s\n', str);
print(figHandle, '-dpng',str); 


fprintf('Will try to write out image %s\n', str_grayImg);
imwrite(uint8(NewG2RGBImg), colormap(gray(256)), str_grayImg, 'png');

fprintf('Algorithm Complete for image %s.... DONE\n', fileName);


%%%%%%%%%%%%% Helper functions %%%%%%%%%%%%%%%%%%%%%%

function [v]=getMagicMatrixValue(M,i,j,N)
if (i == j)
    v = 0;
elseif(i < j)
    v = -1*M(getMagicMatrixIndex(j,i,N));
else v = M(getMagicMatrixIndex(i,j,N));
end;

function [k]= getMagicMatrixIndex(i,j,N)
% k  = ((j - 1)*j/2) + i;  %  for lower diagonal  i <= j
k = i + (((2*N)-j)*(j-1)/2);  % for upper diagonal  i >= j

function [x] = getSizeMagicMatrix(N)
x = 0;
for i=1:N
    x = x+i;
end;

function [Tij] = CalcTij(i,j, LABImgVec, crunchTop)

if (i == j)
    Tij=0;
else 
    
    dL = LABImgVec(1,i)-LABImgVec(1,j);
    da = LABImgVec(2,i)-LABImgVec(2,j);
    db = LABImgVec(3,i)-LABImgVec(3,j);
    C = sqrt((da*da)+(db*db));
    
    %crunch the differences instead...
    C = crunch(C,crunchTop);
    
    
    % If ideally what we want is
    %    if (gL > C)   %where C is the chrominance difference
    %    then gL_NEW =  gL
    %    else  |gL_NEW| = |C| &&  sign(gL_NEW) = sign(gL) 
    
    
    %%Add in contraint to make it keep luminance difference
    %%if chrominance differences are less
    if (abs(dL) > C)
        Tij= dL;
    else
        
        A1 = LABImgVec(2,i);
        A2 = LABImgVec(2,j);
        B1 = LABImgVec(3,i);
        B2 = LABImgVec(3,j);
        
        
        if (i==1 && j==1)
            fprintf('\t\tUsing Theta = 135, Cool(green/blue) to Warm (yellow/red)\n');
        end;
        [YesCool,Mag] = coolColor(A1,B1,A2,B2) ; %% Blue & Green are cool;  Yellow &   Red are warm;
        
        if (YesCool) Tij = C*-1;
        else Tij = C;
        end;
    end;
end;


function [x] = crunch(c,ytop)
%Crunch value c   to range [+ytop, -ytop]
% via the hyperbolic tangent (sigmoid like curve)
% in which small values are preserved but large ones approach ytop
% Crunch(c,ytop) = ytop * tanh(c/ytop)
% creates values which range +/- ytop
x  = ytop * tanh(c/ytop);


