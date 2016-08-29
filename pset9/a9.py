# eseitz - Emily Seitz
# 4/15/12
# 6.815 pset9

import imageIO
from imageIO import *
import numpy
from numpy import *
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


# 2.1 - Scaling using scipy
def scaleChannel(im, i, k):
    scaled = constantIm(height(im), width(im), 0.0)
    channel = im[:, :, i]
    coord = zeros([2, height(im), width(im)])
    # set coord 3D array
    for y, x in imIter(im):
        fy = 1 if (y>height(im)/2) else -1
        fx = 1 if (x>width(im)/2) else -1
        coord[0, y, x] = clipY(im, height(im)/2 + fy*k * abs(y-height(im)/2))
        coord[1, y, x] = clipX(im, width(im)/2 + fx*k * abs(x-width(im)/2))

    # map the channel to the coordinates
    scaled[:, :, i] = scipy.ndimage.map_coordinates(channel, coord)
    for n in range(3):
        if n != i:
            scaled[:, :, n] = im[:, :, n]
    return scaled


# 2.2 - Error function
def lateralChromaError(im, i):
    im[:, :, i] *= mean(im[:, :, 1])/mean(im[:, :, i])
    return sum((im[:,:,i]-im[:,:,1])**2)


# 2.3 - Calibration
def findChroma(im, i, maxPixel=3, steps=20):
    # bestSoFar = (error, scale factor)
    bestSoFar = (1e10, 0)
    step = float(maxPixel)/max(height(im), width(im))
    for s in range(-steps/2, steps/2):
        k = 1 + s*2*step/steps
   # for k in arange(1-step, 1+step+step/steps, 2*step/steps):
        err = lateralChromaError(scaleChannel(im, i, k), i)
        if err < bestSoFar[0]:
            bestSoFar = (err, k)
    return bestSoFar[1]


# 2.4 - Putting it all together
def calibrate(im):
    kr = findChroma(im, 0, 3, 20)
    print "kr = ", kr
    kb = findChroma(im, 2, 3, 20)
    print "kb = ", kb
    return scaleChannel(scaleChannel(im, 0, kr), 2, kb)
    
def testChroma():
    imwriteSeq(scaleChannel(scaleChannel(imread('artiChroma.png'), 0, 1.1), 2, 0.9))


# 2.5 - Calibrate your own lens
def calibrateEmily():
    # calibrate with checkerboard
    im = imread("checkerboard.png")
    kr = findChroma(im, 0, 3, 20)
    print "kr = ", kr
    kb = findChroma(im, 2, 3, 20)
    print "kb = ", kb
    imwrite(scaleChannel(scaleChannel(im, 0, kr), 2, kb), "my_checkerboard.png")

    # apply to other image
    im = imread("ron_psetting.png")
    imwrite(scaleChannel(scaleChannel(im, 0, kr), 2, kb), "ron_psetting.png")
    return
