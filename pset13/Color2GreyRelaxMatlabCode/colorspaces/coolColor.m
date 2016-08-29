function [YesCool,Mag] = coolColor(A1,B1,A2,B2)
%Returns YesCool = 1 if C1 is cooler than C2
% Returns distance in AB between C1 and C2

diffChrominance= sqrt((A2*A2)+(B2*B2)) - sqrt((A1*A1)+(B1*B1));
%//Vector from LAB_1 to LAB_2
AB=[A2-A1  B2-B1];
 ABlength=sqrt((AB(1)*AB(1))+(AB(2)*AB(2)));
%//CoolWarmLine (1,-1) to (-1,1) is the vector (-1,1)
CoolWarmLine=[-1 1];
PerpenLine=[-1 -1];
TempThreshold = .1;

trueG = 0;

 thetaCW = Angle2D(CoolWarmLine(1),CoolWarmLine(2),AB(1),AB(2));

%  x1 = CoolWarmLine(1);
%  y1 = CoolWarmLine(2);
%  x2= AB(1);
%  y2 = AB(2);
% theta1 = atan2(y1,x1)
% theta2 = atan2(y2,x2)
% dtheta = theta2 - theta1

if (ABlength > TempThreshold)
    if (thetaCW > 0)
        trueG = -ABlength;  
        %//make this warm pix brighter
       %% fprintf('\t Should be warm to cool, trueG = %f\n', trueG);
        YesCool = 0;
    elseif (thetaCW < 0)
        trueG = ABlength; 
        %//make this cool pix darker
       %% fprintf('\t Should be cool to warm, trueG = %f\n', trueG);
        YesCool =1;
    else 
        [YesCool Mag] = coolColorRB(A1,B1,A2,B2)
        %%YesCool = 2;
    end;
else
    YesCool  = 2;
end;

Mag = ABlength;

return;

%CIELAB:
%
% +a  =  red
% -a = green
% +b = yellow
% -b = blue
% 	
% 	%blue to red =  cool to warm
% 	coolColor(0,-50,50,0)
% 	
% 	%red to blue
% 	coolColor(50,0,0,-50)
% 	
% 	%yellow to blue
% 	coolColor(0,50,0,-50)
% 	
% 	%blue to yellow  =  cool to warm
% 	coolColor(0,-50,0,50)
% 	
% 	%red to green
% 	coolColor(50,0,-50,0)
% 	
% 	%green to red  =  cool to warm
% 	coolColor(-50,0,50,0)