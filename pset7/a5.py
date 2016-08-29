# eseitz - Emily Seitz
# 3/12/12
# 6.815 A5

import imageIO
from imageIO import *
import numpy
from numpy import *
import scipy
from scipy import *
import a1
from a1 import *
import bilagrid
from bilagrid import *
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


# 2.1 - Weights
def computeWeight(im, epsilonMini=0.002, epsilonMaxi=0.99):
    weights = constantIm(height(im), width(im), 1.0)
    weights[im < epsilonMini] = 0.0
    weights[im > epsilonMaxi] = 0.0
    return weights


# 2.2 - Factor
def computeFactor(im1, w1, im2, w2):
    factor = constantIm(height(im1), width(im1), 0.0)
    ratio = im2 / im1
    ratio = w1 * ratio
    ratio = w2 * ratio
    factor[ratio>0.0] = ratio[ratio>0.0]
    factor = sort(factor.flatten(), None)
    for x in range(len(factor)):
        if factor[x] > 0:
            factor = factor[x:]
            break
    return factor[len(factor)/2]


# 2.3 - Marge to HDR
def makeHDR(imageList, epsilonMini=0.01, epsilonMaxi=0.99):
    weightsList = []
    # save bright pixels of darkest image
    weightsList.append(computeWeight(imageList[0], epsilonMini, 1.01))
    for im in range(1, len(imageList)-1):
        weightsList.append(computeWeight(imageList[im], epsilonMini, epsilonMaxi))
    # save dark pixels of brightest image
    weightsList.append(computeWeight(imageList[-1], -0.01, epsilonMaxi))

    factorsList = [1.0]
    for i in range(1, len(imageList)):
        factorsList.append(factorsList[i-1] * computeFactor(imageList[i-1], weightsList[i-1], imageList[i], weightsList[i]))
    hdr = constantIm(height(imageList[0]), width(imageList[0]), 0.0)
    for im in range(len(imageList)):
        hdr += weightsList[im] * imageList[im] / factorsList[im]
    w = weightsList[0]
    for wi in range(1,len(weightsList)):
        w += weightsList[wi]
    w = clip(w, 1e-6, len(weightsList))
    
    return hdr/w


# 3 - Tone mapping
def toneMap(im, targetBase=100, detailAmp=3, useBila=False):
    lumi, chromi = lumiChromi(im)
    minLumi = 1e-6 # just in case everything is zero
    flatLumi = sort(lumi.flatten(), None)
    for x in flatLumi:
        if x > 0:
            minLumi = x
            break
    lumi = clip(lumi, minLumi, 1.0)
    logLumi = log10(lumi)
    base = bilaBase(logLumi) if useBila else gaussBase(logLumi)

    # scaling factor
    detail = logLumi - base
    # log10(targetBase) = k*(max - min) + b
    # 0 = k*max + b
    logRange = max(base.flatten()) - min(base.flatten())
    k = log10(targetBase) / logRange
    outLog = detailAmp*detail + k*(base - max(base.flatten()))
    outIntensity = 10**outLog

    return outIntensity * chromi

def bilaBase(logLumi):
    bila = bilaGrid(logLumi[:,:,0], max((width(logLumi), height(logLumi)))/50.0, 0.4)
    return bila.doit(logLumi)

def gaussBase(logLumi):
    dev = max((width(logLumi), height(logLumi)))/50.0
    return ndimage.filters.gaussian_filter(logLumi, [dev, dev, dev])


# 4 - Putting it all together
def pipeline(title, num):
    # merge using HDR
    imageList = []
    for i in range(1, num+1):
        imageList.append(imread(title+"-"+str(i)+".png"))
    hdr = makeHDR(imageList)
    imwrite(hdr, title+"_hdr.png")

    # tone mapping
    imwrite(toneMap(hdr, 100, 1, False), title+"_tonemap_gauss.png")
    imwrite(toneMap(hdr, 100, 2, True), title+"_tonemap_bila.png")
    return

def part4(num=3):
    images = [("vine",3), ("design",7),("ante2",2), ("ante1",2), ("ante3",4), ("horse",2), ("nyc",2), ("sea",2), ("stairs",2)]
    for im in range(num):
        print "working on: ", images[im][0]
        pipeline(images[im][0], images[im][1])
    return
    

def test():
    a = imread("ante2-1.png")
    b = imread("ante2-2.png")
    e = makeHDR([a, b])
    return e

def test2():
    a = imread("ante3-1.png")
    b = imread("ante3-2.png")
    c = imread("ante3-3.png")
    d = imread("ante3-4.png")
    e = makeHDR([a, b, c, d])
    return e

def test3():
    a = imread("design-1.png")
    b = imread("design-2.png")
    c = imread("design-3.png")
    d = imread("design-4.png")
    e = imread("design-5.png")
    f = imread("design-6.png")
    g = imread("design-7.png")
    return makeHDR([a, b, c, d, e, f, g])

def test4():
    a = imread("vine-1.png")
    b = imread("vine-2.png")
    c = imread("vine-3.png")
    return makeHDR([a, b, c])
