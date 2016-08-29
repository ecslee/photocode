# eseitz - Emily Seitz
# 5/7/12
# 6.815 pset12

import imageIO
from imageIO import *
import numpy
from numpy import *
import a12Help
from a12Help import *
import random as rnd
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


# 2 - Paintbrush splatting
def brush(out, y, x, color, texture):
    h, w, q = shape(texture)
    ho, wo, q = shape(out)
    if y<h/2 or y>ho-h/2.0 or x<w/2 or x>wo-w/2.0:
        return out
    y1 = y - h/2
    y2 = y1 + h
    x1 = x - w/2
    x2 = x1 + w
    out[y1:y2, x1:x2] = texture[:,:]*color + (1-texture[:,:])*out[y1:y2, x1:x2]
    return out


# 3.1 - Single scale
# 3.2 - Importance sampling with rejection
def singleScalePaint(im, out, importance, texture, size=10, N=1000, noise=0.3):
    h, w, q = shape(im)
    N *= h*w*q/sum(importance)
    texture = scaleImage(texture, float(size)/max(shape(texture)))
    for n in range(N):
        y = rnd.randrange(0, h)
        x = rnd.randrange(0, w)
        if rnd.random() < importance[y, x, 0]:
            brush(out, y, x, (1-noise/2+noise*numpy.random.rand(3))*im[y, x], texture)
    return out


# 3.3 - Two-scale painterly rendering
def painterly(im, N=10000, size=50, noise=0.3):
    h, w, q = shape(im)
    out = constantIm(h, w, 0.0)
    texture = imread("NPR/brush.png")

    # first, coarse pass
    importance = constantIm(h, w, 1.0)
    singleScalePaint(im, out, importance, texture, size, N, noise)

    # second, finer pass
    importance = sharpnessMap(im)
    singleScalePaint(im, out, importance, texture, size/4, N, noise)
    
    return out


# 4.1 - Orientation extraction
def orient(im, tensor):
    eigs = constantIm(shape(im)[0], shape(im)[1], 0.0)
    for y, x in imIter(eigs):
        eig = linalg.eigh(tensor[y, x])
        minEig = argmin(eig[0])
        angle = arctan2([eig[1][minEig][1],0], [eig[1][minEig][0],1])[0] + pi/2.0
        if angle < 0: angle += 2*pi
        eigs[y,x] = (angle)/(2*pi)

    return eigs


# 4.2 - Single-scale
def singleScaleOrientedPaint(im, out, tensor, importance, texture, size, N, noise, nAngles=36):
    h, w, q = shape(im)
    N *= h*w*q/sum(importance)
    texture = scaleImage(texture, float(size)/max(shape(texture)))
    brushes = rotateBrushes(texture, nAngles)
    orientM = orient(im, tensor)
    for n in range(N):
        y = rnd.randrange(0, h)
        x = rnd.randrange(0, w)
        if rnd.random() < importance[y, x, 0]:
            brush(out, y, x, (1-noise/2+noise*numpy.random.rand(3))*im[y, x], brushes[int(orientM[y,x][0]*36)])
    return out


# 4.3 - Two scale
def orientedPaint(im, N=7000, size=50, noise=0.3):
    h, w, q = shape(im)
    out = constantIm(h, w, 0.0)
    texture = imread("NPR/brush.png")
    tensor = computeTensor(im, sigmaG=3, factor=5, debug=False)

    # first, coarse pass
    importance = constantIm(h, w, 1.0)
    singleScaleOrientedPaint(im, out, tensor, importance, texture, size, N, noise)

    # second, finer pass
    importance = sharpnessMap(im)
    singleScaleOrientedPaint(im, out, tensor, importance, texture, size/4, N, noise)
    
    return out



def test():
    texture = imread('NPR/brush.png')
    
##    im = imread('NPR/villeperdue.png')
##    out = constantIm(shape(im)[0], shape(im)[1], 0.0)
##    importance = constantIm(shape(im)[0], shape(im)[1], 1.0)
##    tensor = computeTensor(im, sigmaG=3, factor=5, debug=False)
##    imwriteSeq(singleScaleOrientedPaint(im, out, tensor, importance, texture, 50, 10000, 0.3, 36))

    im = imread('flowers-huge.png')
    out = constantIm(shape(im)[0], shape(im)[1], 0.0)
    importance = constantIm(shape(im)[0], shape(im)[1], 1.0)
    tensor = computeTensor(im, sigmaG=3, factor=5, debug=False)
    imwriteSeq(singleScaleOrientedPaint(im, out, tensor, importance, texture, 40, 10000, 0.3, 36))

def myphotos():
    #imwrite(orientedPaint(imread('sunset.png'), 9000, 40), 'sunset-oriented.png')
    #imwrite(orientedPaint(imread('ring_delivery.png'), 12000, 35), 'ring-delivery-oriented.png')
    #imwrite(orientedPaint(imread('sailboat.png'), 9000, 40), 'sailboat-oriented.png')
    #imwrite(orientedPaint(imread('NPR/archie.png')), 'archie-oriented.png')
    #imwrite(orientedPaint(imread('flowers-huge.png'), 70000, 60), 'flowers-huge-oriented.png')
    imwrite(orientedPaint(imread('the-log.png'), 70000, 60), 'the-log-oriented.png')

