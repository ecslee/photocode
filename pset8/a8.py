# eseitz - Emily Seitz
# 4/9/12
# 6.815 pset8

import imageIO
from imageIO import *
import numpy
from numpy import *
import scipy.ndimage
import scipy.signal
import a6
from a6 import *
import a7
from a7 import *
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# 2.1 - Linear Blending
def linearWeights(im):
    blend = constantIm(height(im), width(im), 0.0)
    for y, x in imIter(blend):
        weightX = (width(im)/2.0-abs(width(im)/2.0 - x))/(width(im)/2.0)
        weightY = (height(im)/2.0-abs(height(im)/2.0 - y))/(height(im)/2.0)
        blend[y, x] = weightX * weightY
    return blend

def linearBlend(im1, im2, bbox, trans, H):
    out = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)

    # image1
    blendIm1 = linearWeights(im1)
    out1a = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    out1b = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    applyhomography(blendIm1, out1a, trans, False)
    applyhomography(im1, out1b, trans, False)

    # image2
    blendIm2 = linearWeights(im2)
    out2a = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    out2b = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    applyhomography(blendIm2, out2a, dot(H, trans), True)
    applyhomography(im2, out2b, dot(H, trans), True)

    # combine
    for y, x in imIter(out):
        denom = clip(out1a[y,x]+out2a[y,x], 1e-6, 1.0)
        out[y,x] = (out1a[y,x]*out1b[y,x]+out2a[y,x]*out2b[y,x])/denom
    return out

def autostitch(im1, im2, twoScale, blurDesc=0.5, radiusDesc=4):
    # compute homography
    features1 = computeFeatures(im1, HarrisCorners(im1), blurDesc, radiusDesc)
    features2 = computeFeatures(im2, HarrisCorners(im2), blurDesc, radiusDesc)
    ransac = RANSAC(findCorrespondences(features1, features2))
    H = ransac[3]

    # compute bounding box and translation matrix
    # use im1 as reference
    bbox2 = computeTransformBBox(im2, linalg.inv(H))
    I = array([array([1, 0, 0]), array([0, 1, 0]), array([0, 0, 1])])
    bbox1 = computeTransformBBox(im1, I)
    bbox = bboxUnion(bbox1, bbox2)
    trans = linalg.inv(translate(bbox))

    # composite images
    out = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    if not twoScale:
        return linearBlend(im1, im2, bbox, trans, H)
    if twoScale:
        return twoScaleBlend(im1, im2, bbox, trans, H)
    return out


# 2.2 - Two-scale blending
def twoScaleFreq(im):
    lumi = constantIm(height(im), width(im), 0.0)
    bw = dot(im, array([0.3, 0.6, 0.1]))
    for y, x in imIter(lumi):
        lumi[y, x] = dot(im[y, x], array([0.3, 0.6, 0.1]))
    lumi = clip(lumi, 1e-6, 1.0)
    chromi = im/lumi
    loFreq = chromi * scipy.ndimage.filters.gaussian_filter(lumi, 2)
    hiFreq = im - loFreq
    return (loFreq, hiFreq)

def twoScaleBlend(im1, im2, bbox, trans, H):
    outLo = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    outHi = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)

    # image1
    im1Lo, im1Hi = twoScaleFreq(im1)
    blend1 = linearWeights(im1)
    #out1Lo = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    out1Hi = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)

    # image2
    im2Lo, im2Hi = twoScaleFreq(im2)
    blend2 = linearWeights(im2)
    #out2Lo = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    out2Hi = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)

    # LOW FREQUENCIES
    # image1
    out1a = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    out1b = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    applyhomography(blend1, out1a, trans, False)
    applyhomography(im1Lo, out1b, trans, False)

    # image2
    out2a = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    out2b = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    applyhomography(blend2, out2a, dot(H, trans), True)
    applyhomography(im2Lo, out2b, dot(H, trans), True)

    # combine
    for y, x in imIter(outLo):
        denom = clip(out1a[y,x]+out2a[y,x], 1e-6, 1.0)
        outLo[y,x] = (out1a[y,x]*out1b[y,x]+out2a[y,x]*out2b[y,x])/denom

    # HIGH FREQUENCIES
    applyhomography(im1Hi, out1Hi, trans, False)
    applyhomography(im2Hi, out2Hi, dot(H, trans), True)
    for y, x in imIter(outHi):
        if out1a[y,x,0] > out2a[y,x,0]:
            outHi[y, x] = out1Hi[y, x]
        else:
            outHi[y, x] = out2Hi[y, x]
    out = outLo + outHi
    return out


# 4 - Extra credit
# Cylindrical reprojection
def cylindrical(im1, im2, f, twoScale, blurDesc=0.5, radiusDesc=4):
    # compute cylindrical components
    K = array([[f, 0, 0], [0, f, 0], [0, 0, 1]])
    size = max(height(im1), width(im1))
    S = array([[size, 0, height(im1)/2.0], [0, size, width(im1)/2.0], [0, 0, 1]])

    # compute homography
    features1 = computeFeatures(im1, HarrisCorners(im1), blurDesc, radiusDesc)
    features2 = computeFeatures(im2, HarrisCorners(im2), blurDesc, radiusDesc)
    ransac = RANSAC(findCorrespondences(features1, features2))
    H = dot(ransac[3], dot(S,K))
    print H

    # compute bounding box and translation matrix
    # use im1 as reference
    bbox2 = computeTransformBBox(im2, linalg.inv(H))
    I = array([array([1, 0, 0]), array([0, 1, 0]), array([0, 0, 1])])
    bbox1 = computeTransformBBox(im1, I*S*K)
    bbox = bboxUnion(bbox1, bbox2)
    bbox = [[bbox[0][0], -pi/2.0], [bbox[1][0], pi/2]]
    trans = linalg.inv(translate(bbox))

    # composite images
    print bbox
    out = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    
    
##    if not twoScale:
##        return linearBlend(im1, im2, bbox, trans, H)
##    if twoScale:
##        return twoScaleBlend(im1, im2, bbox, trans, H)
    return out

# tests
def test():
    a = imread('pano/stata-small-2.png')
    b = imread('pano/stata-small-1.png')
    aC = HarrisCorners(a)
    bC = HarrisCorners(b)
    aF = computeFeatures(a, aC, 0.5, 2)
    bF = computeFeatures(b, bC, 0.5, 2)
    co = findCorrespondences(aF, bF)
    visualizePairs(a, b, co)
    ransac = RANSAC(co)
    out = linearBlend(a, b, ransac[3])

    c = constantIm(height(a), width(a), 1.0)
    d = constantIm(height(b), width(b), 0.7)
    out = linearBlend(c, d, ransac[3])
    imwriteSeq(out, "linear-blend-")
    return

def stata():
    a = imread('pano/stata-1.png')
    b = imread('pano/stata-2.png')
    out = autostitch(a, b, False)
    imwriteSeq(out, "linear-blend-")
    return

def test2():
    #a = imread('pano/stata-small-2.png')
    #b = imread('pano/stata-small-1.png')
    a = imread('pano/stata-1.png')
    b = imread('pano/stata-2.png')
    out = autostitch(a, b, True)
    imwriteSeq(out, "linear-")
    return

def versailles():
    a = imread('versailles3-2.png')
    b = imread('versailles3-3.png')
    out = autostitch(a, b, False)
    imwrite(out, "vesailles-linear.png")
    out = autostitch(a, b, True)
    imwrite(out, "versailles-twoscale.png")
    return

def sunrise():
    a = imread('sunrise1.png')
    b = imread('sunrise2.png')
    out = autostitch(a, b, True)
    imwrite(out, "sunrise-twoscale.png")
    out = autostitch(a, b, False)
    imwrite(out, "sunrise-linear.png")
    return

def testCylin():
    a = imread('pano/stata-small-2.png')
    b = imread('pano/stata-small-1.png')
    #a = imread('versailles3-2.png')
    #b = imread('versailles3-3.png')
    out = cylindrical(a, b, 1, False)
    imwriteSeq(out, "cylindrical-")
    return

