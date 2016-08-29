import imageIO
from imageIO import *
import a2
from a2 import *
import numpy
from numpy import *
import scipy
from scipy import signal
from scipy import ndimage
import a7help
reload(a7help)
from a7help import *
import random as rnd
import math

Sobel=array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

def sharpnessMap(im, sigma=1):
    L=dot(im, array([0.3, 0.6, 0.1]))
    blur=ndimage.filters.gaussian_filter(L, sigma)
    high=L-blur
    energy=high*high
    sharpness=ndimage.filters.gaussian_filter(energy, 4*sigma)
    sharpness/=max(sharpness.flatten())
    return imageFrom1Channel(sharpness)

def computeTensor(im, sigmaG=1, factor=4, debug=False):
    L=dot(im, array([0.3, 0.6, 0.1]))
    L=L**0.5
    L=ndimage.filters.gaussian_filter(L, sigmaG)
    gx=signal.convolve(L, Sobel, mode='same')
    gy=signal.convolve(L, Sobel.T, mode='same')

    h, w=im.shape[0], im.shape[1]
    
    gx[:, 0:2]=0
    gy[0:2, :]=0
    gx[:, w-2:w]=0
    gy[h-2:h, :]=0

    out = empty([L.shape[0], L.shape[1], 2, 2])
    
    out[:, :, 0, 0]=gy*gy
    out[:, :, 0, 1]=gy*gx
    out[:, :, 1, 0]=gy*gx
    out[:, :, 1, 1]=gx*gx

    out=ndimage.filters.gaussian_filter(out, [sigmaG*factor, sigmaG*factor, 0, 0])
    return out

def eigenVec(triplet):
    y,x =1.0, 0.0
    def ap(y, x):
        return triplet[0]*y+triplet[1]*x, triplet[1]*y+triplet[2]*x
    for i in xrange(20):
        y, x=ap(y, x)
        r=sqrt(y*y+x*x)
        y/=r
        x/=r
    return y, x
    

def scaleImage(im, k):
    h, w=int(im.shape[0]*k), int(im.shape[1]*k)
    out = constantIm(h, w, 0.0)
    coord=mgrid[0:h, 0:w]*1.0/k
    for i in xrange(3):
        out[:,:,i]=ndimage.map_coordinates(im[:, :, i], coord, mode='nearest', order=3)
    return out

def rotateImage(im, theta):
    h, w=int(im.shape[0]), int(im.shape[1])
    out = empty_like(im)
    coord=mgrid[0:h, 0:w]*1.0
    ct, st=cos(theta), sin(theta)
    coord2=empty_like(coord)

    coord[0]-=h/2
    coord[1]-=w/2
    coord2[0]=st*coord[1]+ct*coord[0]+h/2
    coord2[1]=ct*coord[1]-st*coord[0]+w/2
    for i in xrange(3):
        out[:,:,i]=ndimage.map_coordinates(im[:, :, i], coord2, mode='nearest', order=3)
    return out

def rotateBrushes(texture, n):
    L=[]
    for i in xrange(n):
        theta=2*math.pi/n*i
        tmp=rotateImage(texture, theta)
        L.append(tmp)
    return L
