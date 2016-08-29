%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  ReadMe for Color2Grey matlab code
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

This code is a small prototype of our system, it doesn't have all of the optimizations or parameters of our complete system.

Additionally, it is implemented for speed of prototyping and using matlab's speed of procesing vectors, which has a side effect of not allowing large images.  (IE a 10x10 image computes fast.. 25x25 takes much, much longer, and you can't use large images due to memory constraints). These are implementation problems, not a problem with the technique.

Usage:

Copy folder:
     Color2GreayRelaxMatlabCode

add Color2GreayRelaxMatlabCode and all is its subdirectories to your matlab path

in matlab,
cd  to the TestImages directory

Color2Grey('Sunrise.png')

It will pop up a figure with the New grayscale image (converted from LAB to RGB for display), the original color images, and the luminance channel from LAB (converted to RGB for display).

The matlab code is commented with the variable you can change.  One important thing to note is that it resizes the image, and the default is currently set pretty small.