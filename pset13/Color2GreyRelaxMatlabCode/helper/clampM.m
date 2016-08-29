function [Mnew] = clampM(M,minv,maxv)
% function [Mnew] = clamp(M,minv,maxv)
% Given a matrix M, checks each value v, this function clamps value to be in range minv to maxv
[m, n] = size(M);
Mnew = zeros(size(M));
for j=1:m
    for i = 1:n
        v = M(j,i);
        
        if (v == NaN)
            val = 0;
        elseif (v > maxv)
            val= maxv;
        elseif(v < minv)
            val  =minv;
        else val = v;
        end;
        Mnew(j,i) = val;
    end;
end;