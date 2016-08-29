# eseitz - Emily Seitz
# 3/19/12
# 6.815 pset 6

import imageIO
from imageIO import *
import numpy
from numpy import *
import a2
from a2 import *
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

def getBlackPadded(im, y, x):
    if (x<0) or (x>=im.shape[1]) or (y<0) or (y>=im.shape[0]):
        return array([0,0,0])
    else:
        return im[y,x]


# 2 - Warp by a homography
def applyhomography(source, out, H, bilinear=False):
    if not bilinear:
        for y, x in imIter(out):
            pixel = dot(H, [y, x, 1])
            if clipY(source, pixel[0]/pixel[2])==pixel[0]/pixel[2] and clipX(source, pixel[1]/pixel[2])==pixel[1]/pixel[2]:
                out[y, x] = source[pixel[0]/pixel[2]][pixel[1]/pixel[2]]
    else:
        for y, x in imIter(out):
            pixel = dot(H, [y, x, 1])
            if clipY(source, pixel[0]/pixel[2])==pixel[0]/pixel[2] and clipX(source, pixel[1]/pixel[2])==pixel[1]/pixel[2]:
                out[y, x] = interpolateLin(source, pixel[0]/pixel[2], pixel[1]/pixel[2])
    return


# 3 - Solve for a homography from four pairs of points
def computehomography(listOfPairs):
    A = []
    B = []
    for pair in listOfPairs:
        y = pair[0][0]
        x = pair[0][1]
        y_ = pair[1][0]
        x_ = pair[1][1]
        A.append([y, x, 1, 0, 0, 0, -y*y_, -x*y_])#, -y_])
        A.append([0, 0, 0, y, x, 1, -y*x_, -x*x_])#, -x_])
        B.append(y_)
        B.append(x_)
    #A.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
    if linalg.det(A) == 0:
        A = [[1, 0, 0, 0, 0, 0, 0, 0],
                   [0, 1, 0, 0, 0, 0, 0, 0],
                   [0, 0, 1, 0, 0, 0, 0, 0],
                   [0, 0, 0, 1, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1, 0, 0, 0],
                   [0, 0, 0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 0, 0, 1]]
    A_inv = linalg.inv(numpy.array(A))
    #B.append(1)
    B = numpy.array(B)

    # A * x = B
    #     x = A_inv * B
    xVec = dot(A_inv, B)
    xVec.resize(9)
    xVec[8] = 1.0
    return reshape(xVec, (3, 3))


# 4.1 - Transform a bounding box
def computeTransformBBox(im, H):
    ymin = 0
    ymax = 0
    xmin = 0
    xmax = 0
    for y, x in imIter(im):
            pixel = dot(H, [y, x, 1])
            if pixel[0]/pixel[2] < ymin: ymin = pixel[0]/pixel[2]
            if pixel[0]/pixel[2] > ymax: ymax = pixel[0]/pixel[2]
            if pixel[1]/pixel[2] < xmin: xmin = pixel[1]/pixel[2]
            if pixel[1]/pixel[2] > xmax: xmax = pixel[1]/pixel[2]
    return array([array([ymin, xmin]), array([ymax, xmax])])


# 4.2 - Bounding box union
def bboxUnion(B1, B2):
    return array([array([floor(min(B1[0][0], B2[0][0])), floor(min(B1[0][1], B2[0][1]))]), array([ceil(max(B1[1][0], B2[1][0])), ceil(max(B1[1][1], B2[1][1]))])])


# 4.3 - Translation
def translate(bbox):
    return array([array([1, 0, -bbox[0][0]]), array([0, 1, -bbox[0][1]]), array([0, 0, 1])])


# 4.4 - Putting it all together
def stitch(im1, im2, listOfPairs):
    # (1) compute homography
    H = computehomography(listOfPairs)

    # (2) compute bounding box and translation matrix
    # use im1 as reference
    bbox2 = computeTransformBBox(im2, linalg.inv(H))
    I = array([array([1, 0, 0]), array([0, 1, 0]), array([0, 0, 1])])
    bbox1 = computeTransformBBox(im1, I)
    bbox = bboxUnion(bbox1, bbox2)
    trans = linalg.inv(translate(bbox))

    # (3) composite images
    out = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    # first, im1 with black background
    applyhomography(im1, out, trans, False)
    # second, im2 with im1/black
    applyhomography(im2, out, dot(H, trans), True)

    return out
    

# 4.5 - N images
def stitchN(listOfImages, listOfListOfPairs, refIndex):
    # pair = im_i and im_i+1
    # (1) compute homographies, H[i] = im[i]&im[i-1]
    I = array([array([1, 0, 0]), array([0, 1, 0]), array([0, 0, 1])])
    listOfH = []
    for i in range(len(listOfImages)):
        listOfH.append(0)
    listOfH[refIndex] = I
    # compute homographies for images with indeces before refIndex
    for i in range(1, refIndex+1):
        pairFlip = []
        for p in listOfListOfPairs[refIndex-i]:
            pairFlip.append([p[1], p[0]])
        listOfH[refIndex-i] = dot(computehomography(pairFlip), listOfH[refIndex-i+1])
    # compute homographies for images with indices after refIndex
    for i in range(refIndex, len(listOfImages)-1):
        listOfH[i+1] = dot(computehomography(listOfListOfPairs[i]), listOfH[i])

    # (2) compute bounding box and translation matrix
    bboxU = computeTransformBBox(listOfImages[0], linalg.inv(listOfH[0]))
    for i in range(1, len(listOfImages)):
        bboxU = bboxUnion(bboxU, computeTransformBBox(listOfImages[i], linalg.inv(listOfH[i])))
    trans = linalg.inv(translate(bboxU))

    # (3) composite images
    out = constantIm(bboxU[1][0]-bboxU[0][0], bboxU[1][1]-bboxU[0][1], 0.0)
    for i in range(len(listOfImages)):
        applyhomography(listOfImages[i], out, dot(listOfH[i], trans), True)

    return out

