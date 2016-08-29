function [dtheta]= Angle2D(x1,y1,x2,y2)
%/*  http://astronomy.swin.edu.au/~pbourke/geometry/insidepoly/
%    Return the angle between two vectors on a plane
%    The angle is from vector 1 to vector 2, positive anticlockwise
%    The result is between -pi -> pi
% */
theta1 = atan2(y1,x1);
theta2 = atan2(y2,x2);
dtheta = theta2 - theta1;
while (dtheta > pi)
    dtheta = dtheta - (pi*2.0);  %//TWOPI;
end;
while (dtheta < -pi)
    dtheta = dtheta + (pi*2.0); %//TWOPI;
end;
