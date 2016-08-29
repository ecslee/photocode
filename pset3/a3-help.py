import imageIO
from imageIO import *
import a3
import numpy
from numpy import *

def impulse(h=100, w=100):
    out=constantIm(h, w, 0.0)
    out[h/2, w/2]=1
    return out

box3=ones((3, 3))/9.0
gauss3=array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

deriv=array([[-1, 1]])

Sobel=array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

def addNoise(im, mu=0, sigma=0.1):
    noise =  numpy.random.normal(mu, sigma, im.shape)
    return im+noise
