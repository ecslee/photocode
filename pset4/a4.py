# eseitz - Emily Seitz
# 3/5/12
# 6.815 A4

import imageIO
from imageIO import *
import numpy
from numpy import *
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

def getBlackPadded(im, y, x):
    if (x<0) or (x>=im.shape[1]) or (y<0) or (y>=im.shape[0]):
        return numpy.array([0,0,0])
    else:
        return im[y,x]


# 2.1 - Basic sequence denoising
def denoiseSeq(imageList):
    denoised = constantIm(height(imageList[0]), width(imageList[0]), 0.0)
    for im in imageList:
        denoised += im
    denoised /= len(imageList)
    return denoised

##    for y, x in imIter(denoised):
##        val = 0
##        for im in imageList:
##            val += im[y][x]
##        val /= len(imageList)
##        denoised[y, x] = val
##    return denoised


# 2.2 - Variance
def logSNR(imageList, scale=1.0/20.0):
    mean = denoiseSeq(imageList)
    varImList = []
    for im in imageList:
        varImList.append((mean - im)**2)
    variance = clip(denoiseSeq(varImList), 1e-6, 1.0)
    exp_squared = variance + mean**2
    return scale * log10(exp_squared/variance)

# make a list of images
def makeImList(name, n):
    imageList = []
    for i in range(1, n+1):
        imageList.append(imread(name+'-'+str(i)+'.png'))
    return imageList


# 2.3 - Alignment
def align(im1, im2, maxOffset=20):
    best = [1000000, [], 0, 0]
    #tests = []
    for i in range(-maxOffset, maxOffset+1):
        for j in range(-maxOffset, maxOffset+1):
            test = constantIm(height(im1), width(im1), 0.0)
            for y in range(maxOffset, height(test)-maxOffset):
                for x in range(maxOffset, width(test)-maxOffset):
                    test[y][x] = im2[y+i][x+j]
            # compare this one?  if better, save it
            diff = sum((test-im1)**2)
            if diff < best[0]:
                best = [diff, test, i, j]
    return best

def alignAndDenoise(imageList, maxOffset=20):
    alignList = [imageList[0]]
    for im in range(1, len(imageList)):
        alignList.append(align(imageList[0], imageList[im])[1])
    denoised = denoiseSeq(alignList)
    
    # edges of denoised should just copy the original
    # top edge
    for n in range(maxOffset):
        denoised[n] = imageList[0][n]
    # bottom edge
    for n in range(height(imageList[0])-maxOffset, height(imageList[0])):
        denoised[n] = imageList[0][n]
    # right side
    for m in range(height(imageList[0])):
        for n in range(maxOffset):
            denoised[m][n] = imageList[0][m][n]
    # left side
    for m in range(height(imageList[0])):
        for n in range(width(imageList[0])-maxOffset, width(imageList[0])):
            denoised[m][n] = imageList[0][m][n]

    return denoised


# 3.1 - Basic green channel
def basicGreen(raw, offset=1):
    green = []
    for h in range(height(raw)):
        green.append(numpy.zeros(width(raw)))
    green = numpy.array(green)
    for y in range(1, height(green)-1):
        if y%2==0 and offset==1: lo = 1# next = x+1 = x+(-1)**0
        elif y%2==0 and offset==0: lo = 0# previous = x-1 = x+(-1)**1
        elif y%2==1 and offset==1: lo = 0# next = x+1
        elif y%2==1 and offset==0: lo = 1# previous = x-1

        for x in range(2-lo, width(green)-1, 2):
            # copy green for this pixel
            green[y, x] = raw[y, x]

            # average 4 neighbors for the next/previous pixel
            g = 0
            if x+2 <= width(raw):
                for n in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    g += .25 * raw[y+n[0]][x+(-1)**(lo-1)+n[1]]
            green[y][x+(-1)**(lo-1)] = g

    return green


# 3.2 - Basic red and blue
def basicRorB(raw, offsety, offsetx):
    redBlue = []
    for h in range(height(raw)):
        redBlue.append(numpy.zeros(width(raw)))
    redBlue = numpy.array(redBlue)
    for y in range(offsety, height(redBlue)-1, 2):
        for x in range(offsetx, width(redBlue)-1, 2):
            # copy red/blue for this pixel
            redBlue[y][x] = raw[y][x]
            
            # average 2 neighbors for next pixel
            rb = 0
            if x+2 < width(raw):
                rb += .5 * (raw[y][x] + raw[y][x+2])
            redBlue[y][x+1] = rb

            # average 2 neighbors for pixel below
            rb = 0
            if y+2 < height(raw):
                rb += .5 * raw[y][x] + .5 * raw[y+2][x]
            redBlue[y+1][x] = rb

            # average 4 neighbors for next pixel below
            rb = 0
            if x+2 < width(raw) and y+2 < height(raw):
                for n in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
                    rb += .25 * raw[y+1+n[0]][x+1+n[1]]
            redBlue[y+1][x+1] = rb
        # force leading edges to be black
        redBlue[y][0] = 0.0
        redBlue[y+1][0] = 0.0
    redBlue[0] = 0.0

    return redBlue

def basicDemosaick(raw, offsetGreen=1, offsetRedY=1, offsetRedX=1, offsetBlueY=0, offsetBlueX=0):
    demosaick = constantIm(height(raw), width(raw), 0.0)
    demosaick[:, :, 1] = basicGreen(raw, offsetGreen)
    demosaick[:, :, 0] = basicRorB(raw, offsetRedY, offsetRedX)
    demosaick[:, :, 2] = basicRorB(raw, offsetBlueY, offsetBlueX)
    return demosaick


# 4 - Edge-based green
def edgeBasedGreen(raw, offset=1):
    green = []
    for h in range(height(raw)):
        green.append(numpy.zeros(width(raw)))
    green = numpy.array(green)
    for y in range(1, height(green)-1):
        if y%2==0 and offset==1: lo = 1# next = x+1 = x+(-1)**0
        elif y%2==0 and offset==0: lo = 0# previous = x-1 = x+(-1)**1
        elif y%2==1 and offset==1: lo = 0# next = x+1
        elif y%2==1 and offset==0: lo = 1# previous = x-1

        for x in range(2-lo, width(green)-1, 2):
            # copy green for this pixel
            green[y, x] = raw[y, x]

            # average 2 neighbors for the next pixel
            g = 0
            if x+2 <= width(raw):
                up = raw[y-1][x+(-1)**(lo-1)]
                down = raw[y+1][x+(-1)**(lo-1)]
                left = raw[y][x+(-1)**(lo-1)-1]
                right = raw[y][x+(-1)**(lo-1)+1]
                if (up-down)**2 < (left-right)**2:
                    g = .5 * (up + down)
                else:
                    g = .5 * (left + right)
            green[y][x+(-1)**(lo-1)] = g

    return green


# 5 - Red and blue based green
def greenBasedRorB(raw, green, offsety, offsetx):
    redBlue = []
    for h in range(height(raw)):
        redBlue.append(numpy.zeros(width(raw)))
    redBlue = numpy.array(redBlue)
    for y in range(offsety, height(redBlue)-1, 2):
        for x in range(offsetx, width(redBlue)-1, 2):
            # copy red/blue for this pixel
            redBlue[y][x] = raw[y][x]
      #  redBlue[y][0] = 0.0
     #   redBlue[y+1][0] = 0.0
    #redBlue[0] = 0.0
            
    redBlue = redBlue - green
    for y in range(offsety, height(redBlue)-1, 2):
        for x in range(offsetx, width(redBlue)-1, 2):
            # average 2 neighbors for next pixel
            rb = 0
            if x+2 < width(raw):
                rb += .5 * (raw[y][x] - green[y][x])
                rb += .5 * (raw[y][x+2] - green[y][x+2])
            redBlue[y][x+1] = rb

            # average 2 neighbors for pixel below
            rb = 0
            if y+2 < height(raw):
                rb += .5 * (raw[y][x] - green[y][x])
                rb += .5 * (raw[y+2][x] - green[y+2][x])
            redBlue[y+1][x] = rb

            # average 4 neighbors for next pixel below
            rb = 0
            if x+2 < width(raw) and y+2 < height(raw):
                for n in [(-1, -1), (-1, 1), (1, 1), (1, -1)]:
                    rb += .25 * (raw[y+1+n[0]][x+1+n[1]] - green[y+1+n[0]][x+1+n[1]])
            redBlue[y+1][x+1] = rb
        # force leading edges to be black
        redBlue[y][0] = 0.0
        redBlue[y+1][0] = 0.0
    redBlue[0] = 0.0
    
    return redBlue + green

def greenBasedDemosaick(raw, offsetGreen=1, offsetRedY=1, offsetRedX=1, offsetBlueY=0, offsetBlueX=0):
    demosaick = constantIm(height(raw), width(raw), 0.0)
    green = edgeBasedGreen(raw, offsetGreen)
    demosaick[:, :, 1] = green
    demosaick[:, :, 0] = greenBasedRorB(raw, green, offsetRedY, offsetRedX)
    demosaick[:, :, 2] = greenBasedRorB(raw, green, offsetBlueY, offsetBlueX)
    return demosaick

