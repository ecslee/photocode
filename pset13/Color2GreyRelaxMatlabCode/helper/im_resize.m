function newimg = im_resize(img,nw,nh)
%IM_RESIZE Resize an image using bicubic interpolation
%
%          NEWIMG = IM_RESIZE(IMG,NW,NH) Given input image IMG,
%          returns a new image NEWIMG of size NWxNH.

% Matthew Dailey 2000

  if nargin ~= 3
    error('usage: im_resize(image,new_wid,new_ht)');
  end;
  
  ht_scale = size(img,1) / nh;
  wid_scale = size(img,2) / nw;
  
  newimg = interp2(img,(1:nw)*wid_scale,(1:nh)'*ht_scale,'cubic');
