# eseitz - Emily Seitz
# 2/27/12
# 6.815 A3

import imageIO
from imageIO import *
import a3
import numpy
from numpy import *
import math
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Utility functions:
def height(im):
    return im.shape[0]

def width(im):
    return im.shape[1]

def imIter(im):
    for y in xrange(height(im)):
        for x in xrange(width(im)):
            yield y, x

def clipX(im, x):
    return min(width(im)-1, max(x, 0))

def clipY(im, y):
    return min(height(im)-1, max(y, 0))

def getSafePix(im, y, x):
    return im[clipY(im, y), clipX(im, x)]


# 2.1 - Box blur
def boxBlur(im, k):
    blurred = constantIm(height(im), width(im), 0.0)
    # if k is odd, center around the current pixel
    if k%2:
        lo = -(k/2)
        hi = k/2 + 1
    # if k is even, center such that the current pixel is NW of center
    else:
        lo = -(k/2) + 1
        hi = k/2 + 1
        
    for y, x, in imIter(blurred):
        average = 0
        for h in range(lo, hi):
            for w in range(lo, hi):
                average += getSafePix(im, y+h, x+w)
        blurred[y, x] = average / float(k*k)
    return blurred


# 2.2 - General kernel
def convolve(im, kernel):
    convolved = constantIm(height(im), width(im), 0.0)
    half_h = height(kernel)/2
    half_w = width(kernel)/2

    # kernel[h][w] --> im[y+(h-height/2)][x+(w-width/2)]
    for y, x in imIter(convolved):
        c = 0
        for h in range(height(kernel)):
            for w in range(width(kernel)):
                c += kernel[h][w] * getSafePix(im, y+(h-half_h), x+(w-half_w))
        convolved[y, x] = c
    return convolved


# 2.3 - Gradient
def gradientMagnitude(im):
    Sobel=array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horiz = a3.convolve(im, Sobel)
    vert = a3.convolve(im, transpose(Sobel))
    return numpy.sqrt(horiz**2 + vert**2)


# 2.4 - Horizontal gaussian filtering
def dist(pA, pB):
    add = 0
    for p in range(len(pA)):
        add += (pA[p] - pB[p])**2
    return math.sqrt(add)

def horiGaussKernel(sigma, truncate=3):
    size = int(2*ceil(sigma*truncate)+1)
    gauss = numpy.array([zeros(size)])
    for w in range(size):
        gauss[0][w] = exp(-(w - size/2)**2 / float(2 * sigma**2))
    return gauss / float(sum(gauss))


# 2.5 - Separable gaussian filtering
def gaussianBlur(im, sigma, truncate=3):
    gauss = horiGaussKernel(sigma, truncate)
    horiz = a3.convolve(im, gauss)
    #imwrite(horiz, 'pru-h-gauss-emily.png')
    #imwrite(100*(imread('pru-h-gauss.png')-horiz), 'pru-h-gauss-diff.png')
    vert = a3.convolve(horiz, transpose(gauss))
    return vert


# 2.6 - Verify separability and its usefulness
def gauss2D(sigma=2, truncate=3):
    h = horiGaussKernel(2, 3)
    return transpose(h) * h

def timeGauss():
    im = imread('pru3.png')
    t = time.time()
    a = gaussianBlur(im, 2, 3)
    print "1D: ", time.time() - t, " seconds"
    imwrite(a, 'pru-g-1d-emily.png')
    imwrite(100*(imread('pru-g.png')-a), 'pru-g-1d-diff.png')

    t = time.time()
    b = a3.convolve(im, gauss2D(2, 3))
    print "2D: ", time.time() - t, " seconds"
    imwrite(b, 'pru-g-2d-emily.png')
    imwrite(100*(imread('pru-g.png')-b), 'pru-g-2d-diff.png')
    return


# 2.7 - Sharpening
def unsharpMask(im, sigma, strength):
    sharp = constantIm(height(im), width(im), 0.0)
    gauss = gaussianBlur(im, sigma)
    imwriteSeq(gauss)
    return im + strength*(im - gauss)


# 3 - Denoising using bilateral filtering
def gaussians(im, sigmaRange, sigmaDomain, y, x, y_, x_):
    g_points = exp(-dist((y,x), (y_,x_))**2 / float(2*sigmaDomain**2))
    g_images = exp(-dist(getSafePix(im,y,x), getSafePix(im,y_,x_))**2 / float(2*sigmaRange**2))
    return g_points * g_images

def bilateral(im, sigmaRange, sigmaDomain):
    truncate = 3
    denoised = constantIm(height(im), width(im), 0.0)
    for y, x in imIter(denoised):
        k = 0
        bila = 0
        for y_ in range(y-truncate*sigmaDomain, y+truncate*sigmaDomain+1):
            for x_ in range(x-truncate*sigmaDomain, x+truncate*sigmaDomain+1):
                g = gaussians(im, sigmaRange, sigmaDomain, y, x, y_, x_)
                k += g
                bila += g * getSafePix(im,y_,x_)
        denoised[y][x] = (1.0/k) * bila
    return denoised


