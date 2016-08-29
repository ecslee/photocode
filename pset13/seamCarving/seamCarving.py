# Seam Carving

import imageIO
from imageIO import *
import numpy
from numpy import *
import a3
from a3 import *
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


# Seam Carving

def carveSeam(im, seam):
    final = constantIm(height(im), width(im)-1, 0.0)
    for p in seam:
        im[p[0], p[1]] = [1.0, 0, 0]

    for p in seam:
        a = im[p[0]].flatten()
        a = delete(a, p[1]*3+2)
        a = delete(a, p[1]*3+1)
        a = delete(a, p[1]*3)
        a = a.reshape(1, width(im)-1, 3)
        final[p[0]] = a[0]
        
    return final

def computeEnergy(im):
    bw = BW(im)
    Sobel = array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    horiz = a3.convolve(bw, Sobel)
    vert = a3.convolve(bw, transpose(Sobel))
    return sqrt(horiz**2 + vert**2)

def minEnergy(E):
    h, w = shape(E)[0:2]
    # initiate cost as energy values
    cost = E[:,:,0]
    # find minimum cost to get to each pixel
    for y in range(1,h):
        for x in range(w):
            min_cost = (1e6, (0,0))
            D = [(-1,-1), (-1,0), (-1,1)] # directions to test
            if x==0: D.remove((-1,-1))
            elif x==w-1: D.remove((-1,1))
            test_costs = []
            for d in D:
                test_costs.append(cost[y+d[0], x+d[1]])
            cost[y, x] += min(test_costs)

    # backtrace minimum cost pat
    start_x = argmin(cost[h-1,:])
    seam = [(h-1, start_x)]
    y = seam[-1][0]
    while y > 0:
        x = seam[-1][1]
        D = [(-1,-1), (-1,0), (-1,1)] # directions to test
        if x==0: D.remove((-1,-1))
        elif x==w-1: D.remove((-1,1))
        test_costs = {}
        for d in D:
            test_costs[cost[y+d[0], x+d[1]]] = d
        best = test_costs[min(test_costs)]
        seam.append((y+best[0], x+best[1]))
        y -= 1

    return seam


def resize(im, final_h, final_w):    
    orig_h, orig_w = shape(im)[0:2]
    # assume horizontal change, so vertical seams
    normal = True
    
    if final_h < orig_h:
        normal = False
        print "vertical change"
        im = transpose(im, [0,1])
        imwriteSeq(im, "trans-")
    
    N = orig_w - final_w # steps of change
    print "N = ", N

    for n in range(N):
        E = computeEnergy(im)
        S = minEnergy(E)
        im = carveSeam(im, S)

    if normal == False:
        print "fliiping back"
        im = transpose(im)
    return im


def BW(im, weights=[0.3, 0.6, 0.1]):
    bw = constantIm(height(im), width(im), 0.0)
    for y, x in imIter(bw):
        bw[y, x] = dot(im[y, x], weights)
    return bw

def test():
    #imwriteSeq(resize(imread('tiny.png'), 133, 198))
    #imwriteSeq(resize(imread('tiny.png'), 13, 64))
    imwriteSeq(resize(imread('copley.png'), 340, 500))
    imwriteSeq(resize(imread('parkstreet.png'), 340, 500))

