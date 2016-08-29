# eseitz - Emily Seitz
# 5/2/12
# 6.815 pset11

import imageIO
from imageIO import *
import numpy
from numpy import *
import a11Help
from a11Help import *
import scipy.ndimage
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Utility functions:
def height(im):
    return im.shape[0]

def width(im):
    return im.shape[1]

def imIter(im):
    for y in xrange(height(im)):
        for x in range(width(im)):
            yield y, x

def clipX(im, x):
    return min(width(im)-1, max(x, 0))

def clipY(im, y):
    return min(height(im)-1, max(y, 0))

def getSafePix(im, y, x):
    return im[clipY(im, y), clipX(im, x)]


# 2.1 - Naive compositing
def naiveComposite(bg, fg, mask, y, x):
    h, w, q = shape(fg)
    bg[y:y+h, x:x+w] = mask[:,:]*fg[:,:] + (1-mask[:,])*bg[y:y+h, x:x+w]
    return bg


# 3.2 - Image dot product and convolution
def dotIm(im1, im2):
    dotProd = zeros(shape(im1)[:2])
    for y, x in imIter(dotProd):
        dotProd[y,x] = dot(im1[y,x], im2[y,x])
    return sum(dotProd)


# 3.3 - Gradient descent
def Poisson(bg, fg, mask, niter=200):
    h, w, q = shape(fg)
    b = Laplacian(fg)
    x = constantIm(h, w, 0.0)
    x = (1-mask)*bg
    for i in range(niter):
        r = b - Laplacian(x)
        r *= mask
        a = dotIm(r, r) / dotIm(r, Laplacian(r))
        x += a*r
    return x


# 3.4 - Conjugate gradient
def PoissonCG(bg, fg, mask, niter):
    h, w, q = shape(fg)
    b = Laplacian(fg)
    x = constantIm(h, w, 0.0)
    x = (1-mask)*bg # x_0
    r = b - Laplacian(x) # r_0
    r *= mask
    d = r # d_0
    for i in range(niter):
        a = dotIm(r, r) / dotIm(d, Laplacian(d))
        x += a*d # x_i+1
        r_ = r - a*Laplacian(d) # r_i+1
        B = dotIm(r_, r_) / dotIm(r, r)
        d = r_ + B*d # d_i+1
        d *= mask
        r = r_ # r_i+1
        r *= mask
    return x


def test():
    bg = imread('/Poisson/waterpool.png')
    fg = imread('/Poisson/bear.png')
    mask = imread('/Poisson/mask.png')
    




