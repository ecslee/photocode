# eseitz - Emily Seitz
# 4/2/12
# 6.815 pset 7

import imageIO
from imageIO import *
import numpy
from numpy import *
import scipy.ndimage
import scipy.signal
import a1
from a1 import *
import a6
from a6 import *
import a7help
from a7help import *
import random as rnd
import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 2.1 - Structure tensor
def computeTensor(im, sigmaG=0.5, factorSigma=4):
    # calculate gradients
    lumi = scipy.ndimage.filters.gaussian_filter(lumiChromi(im)[0][:,:,0], 1.0)**0.5
    Sobel = array([[-1,0,1], [-2,0,2], [-1,0,1]])
    Lx = scipy.signal.convolve(lumi, Sobel, mode="same")
    Ly = scipy.signal.convolve(lumi, transpose(Sobel), mode="same")

    # calculate per-pixel contribution
    tensor = []
    Lx2 = scipy.ndimage.filters.gaussian_filter(Lx*Lx, sigmaG*factorSigma)
    Ly2 = scipy.ndimage.filters.gaussian_filter(Ly*Ly, sigmaG*factorSigma)
    Lxy = scipy.ndimage.filters.gaussian_filter(Lx*Ly, sigmaG*factorSigma)
    for y in range(height(im)):
        row = []
        for x in range(width(im)):
            #lx = Lx[y, x]
            #ly = Ly[y, x]
            #m = array([[lx*lx, lx*ly], [lx*ly, ly*ly]])
            #row.append(scipy.ndimage.filters.gaussian_filter(m, sigmaG*factorSigma))
            row.append(array([[Lx2[y,x], Lxy[y,x]], [Lxy[y,x], Ly2[y,x]]]))
        tensor.append(row)
    tensor = array(tensor)

    # image to save
    tensorIm = constantIm(height(im), width(im), 0.0)
    for y, x in imIter(tensorIm):
        tensorIm[y, x, 0] = tensor[y,x,0,0]
        tensorIm[y, x, 1] = tensor[y,x,0,1]
        tensorIm[y,x,2] = tensor[y,x,1,1]
    #imwriteSeq(tensorIm, "tensor-")
    
    # image to save
    #tensorIm = constantIm(height(im), width(im), 0.0)
    #tensorIm[:,:,0] = scipy.ndimage.filters.gaussian_filter(Lx*Lx, sigmaG*factorSigma)
    #tensorIm[:,:,1] = scipy.ndimage.filters.gaussian_filter(Lx*Ly, sigmaG*factorSigma)
    #tensorIm[:,:,2] = scipy.ndimage.filters.gaussian_filter(Ly*Ly, sigmaG*factorSigma)
    #imwriteSeq(tensorIm, "tensor-")

    return array(tensor)


# 2.2 - Harris corners
def HarrisCorners(im, k=0.15, sigmaG=0.5, factor=4, maxDiam=7, boundarySize=5):
    tensor = computeTensor(im)
    
    # corner response
    corners = constantIm(height(im), width(im), 0.0)
    for y, x in imIter(corners):
        corners[y, x] = linalg.det(tensor[y, x]) - k*trace(tensor[y, x])**2
    #imwriteSeq(corners, "cornerRes-")

    # non-maximum suppression
    corners2 = scipy.ndimage.filters.maximum_filter(corners, maxDiam)
    #imwriteSeq(corners2, "maxDiam-")

    # removing boundary corners
    for y in range(boundarySize):
        corners2[y, :] = 0.0
        corners2[height(corners)-1-y, :] = 0.0
    for x in range(boundarySize):
        corners2[:, x] = 0.0
        corners2[:, width(corners)-1-x] = 0.0

    # putting it all together
    cornersList = []
    for y, x in imIter(corners2):
        if corners2[y,x,0]==corners[y,x,0] and corners2[y,x,1]==corners[y,x,1] and corners2[y,x,2]==corners[y,x,2]:
            cornersList.append(array([y, x]))
    #visualizeCorners(im, cornersList)
    return cornersList


# 3.1 - Descriptors
def computeFeatures(im, cornerL, sigmaBlurDescriptor=0.5, radiusDescriptor=4):
    blur = scipy.ndimage.filters.gaussian_filter(lumiChromi(im)[0][:,:,0], sigmaBlurDescriptor)
    featuresList = []
    for P in cornerL:
        featuresList.append(descriptor(blur, P, radiusDescriptor))
    #visualizeFeatures(featuresList, radiusDescriptor, im)
    return featuresList

def descriptor(blurredIm, P, rD):
    patch = []
    patch = blurredIm[P[0]-rD:P[0]+rD+1, P[1]-rD:P[1]+rD+1]
    patch -= mean(patch)
    patch *= 1/std(patch)
    return (P, patch)


# 3.2 - Best match and 2nd best match test
def findCorrespondences(listFeatures1, listFeatures2, threshold=1.7):
    correspondences = []
    # f1 = (point in listFeatures1, patch around this point)
    for f1 in listFeatures1:
        dist = []
        for f2 in listFeatures2:
            dist.append(sum((f1[1]-f2[1])**2))
        # best = (min distance, point in listFeatures2)
        best = (min(dist), listFeatures2[argmin(dist)][0])
        dist.remove(min(dist))
        secondBest = (min(dist), listFeatures2[argmin(dist)][0])
        if abs(secondBest[0]/best[0]) >= threshold:
            correspondences.append((f1[0], best[1]))
    return correspondences


# 4 - RANSAC
def RANSAC(listOfCorrespondences, Niter=1000, epsilon=4):
    # bestSoFar = [# inliers, pairs, wellDef, homography]
    bestSoFar = [-1, 0, 0, 0]
    for n in range(Niter):
        # get 4 random pairs
        pairs = []
        for i in range(4):
            pairs.append(listOfCorrespondences[rnd.randint(0, len(listOfCorrespondences)-1)])
        
        # compute homography
        H = computehomography(pairs)

        # test homography
        wellDef = []
        for p in listOfCorrespondences:
            mag = array([p[1][0], p[1][1], 1]) - dot(H, array([p[0][0], p[0][1], 1]))
            if sqrt(dot(mag, mag)) <= epsilon:
                wellDef.append(True)
            else:
                wellDef.append(False)

        # is it better?
        if wellDef.count(True) > bestSoFar[0]:
            bestSoFar = [wellDef.count(True), pairs, wellDef, H]

    return bestSoFar


# 5.1 - Automatic panorama for a pair of images
def autostitch(im1, im2, blurDesc=0.5, radiusDesc=4):
    features1 = computeFeatures(im1, HarrisCorners(im1), blurDesc, radiusDesc)
    features2 = computeFeatures(im2, HarrisCorners(im2), blurDesc, radiusDesc)
    ransac = RANSAC(findCorrespondences(features1, features2))
    H = ransac[3]
    visualizePairsWithInliers(im1, im2, findCorrespondences(features1, features2), ransac[2])

    # compute bounding box and translation matrix
    # use im1 as reference
    bbox2 = computeTransformBBox(im2, linalg.inv(H))
    I = array([array([1, 0, 0]), array([0, 1, 0]), array([0, 0, 1])])
    bbox1 = computeTransformBBox(im1, I)
    bbox = bboxUnion(bbox1, bbox2)
    trans = linalg.inv(translate(bbox))

    # composite images
    print bbox
    out = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    # first, im1 with black background
    applyhomography(im1, out, trans, False)
    # second, im2 with im1/black
    applyhomography(im2, out, dot(H, trans), True)

    return out
    

def test():
    a = imread("versailles4-2.png")
    b = imread("versailles4-3.png")
    out = autostitch(a, b)
    imwrite(out, "versailles4-autostitch.png")
    return
    
        

