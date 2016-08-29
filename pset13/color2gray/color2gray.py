# convert to CIE L*a*b*
# http://www.mathworks.com/products/image/examples.html?file=/products/demos/shipping/images/ipexfabric.html#2

import imageIO
from imageIO import *
import numpy
from numpy import *
import a11
from a11 import *
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


# ==  Convert image to CIE L*a*b*  ==
# identify different colors using CIE L*a*b* color space
# L* = luminosity
# a* = where color falls on red-green axis
# b* = where color falls on blue-yellow axis

# http://brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html
def RGB2XYZ(im):
    xyz = constantIm(height(im), width(im), 0.0)
    M = matrix([[0.4124564, 0.3575761, 0.1804375],
                [0.2126729, 0.7151522, 0.0721750],
                [0.0193339, 0.1191920, 0.9503041]])
    for y, x in imIter(xyz):
        xyz[y, x] = dot(M, im[y, x])
    return xyz

def XYZ2RGB(im):
    rgb = constantIm(height(im), width(im), 0.0)
    M = matrix([[ 3.24045484, -1.53713885, -0.49853155],
                [-0.96926639,  1.87601093,  0.04155608],
                [ 0.05564342, -0.20402585,  1.05722516]])
    for y, x in imIter(rgb):
        rgb[y, x] = dot(M, im[y, x])
    return rgb

# http://www.easyrgb.com/index.php?X=MATH&H=07#text7
def XYZ2CIELAB(im):
    epsilon = 0.008856
    ref_X = 95.047
    ref_Y = 100.000
    ref_Z = 108.883

    cielab = constantIm(height(im), width(im), 0.0)
    for y, x in imIter(cielab):
        var_X, var_Y, var_Z = im[y, x] / array([ref_X, ref_Y, ref_Z])
        for var in [var_X, var_Y, var_Z]:
            var = var**(1.0/3) if var>epsilon else 7.787*var+(16.0/116)
        cielab[y, x] = [116.0*var_Y-16, 500.0*(var_X-var_Y), 200.0*(var_Y-var_Z)]
    return cielab

def CIELAB2XYZ(im):
    epsilon = 0.008859
    ref_X = 95.047
    ref_Y = 100.000
    ref_Z = 108.883

    xyz = constantIm(height(im), width(im), 0.0)
    for y, x in imIter(xyz):
        var_Y = (im[y, x, 0] + 16)/116.0
        var_X = im[y, x, 1]/500.0 + var_Y
        var_Z = var_Y - im[y, x, 2]/200.0
        for var in [var_X, var_Y, var_Z]:
            var = var**3 if var**3>epsilon else (var-16.0/116)/7.787
        xyz[y, x] = [ref_X*var_X, ref_Y*var_Y, ref_Z*var_Z]
    return xyz
    

def testCIELAB():
    im = imread('ColorsOrangeS.png')
    print "original ", im[0,0]
    print 'same?    ', XYZ2RGB(CIELAB2XYZ(XYZ2CIELAB(RGB2XYZ(im))))[0,0]

def crunch(x, a):
    if a==0: return 0
    return a * tanh(x/a)


def color2gray(im, radius=True, max_offset=10, theta=45):
    h, w, q = shape(im)
    gray = constantIm(h, w, 0.0)
    if radius==True: radius = max(h, w)

    # first, convert color image to perceptually uniform color space
    cielab = XYZ2CIELAB(RGB2XYZ(im))
    L = cielab[:, :, 0]
    a = cielab[:, :, 1]
    b = cielab[:, :, 2]

    # coords:
    # pixel_i = pixel @ row*width + column

    # second, compute color differences in order to combine
    # luminance and chrominance differences
    pixels_i = zeros([h*w, 2])
    for y, x in imIter(im):
        pixels_i[y*w + x, 0] = y
        pixels_i[y*w + x, 1] = x
    pixels_j = zeros([h*w, 2])
    for y, x in imIter(im):
        pixels_j[y*w + x, 0] = y
        pixels_j[y*w + x, 1] = x

    sigmas = zeros([h*w+w, h*w+w])

    for i in pixels_i:
        for j in pixels_j:
            if max(abs(i[1]-j[1]), abs(i[0]-j[0])) < radius+1:
                y, x = i
                y_, x_ = j
                delta_L = L[y,x] - L[y_,x_]
                delta_C = [a[y,x]-a[y_,x_], b[y,x]-b[y_,x_]]
                cr = crunch(sqrt(delta_C[0]**2 + delta_C[1]**2), max_offset)
                sigma = delta_L
                if abs(delta_L) < cr:
                    d = dot(array(delta_C), array([cos(theta), sin(theta)]))
                    sign = d/abs(d)
                    sigma = sign * cr
                sigmas[y*w+x, y_*w+x_] = sigma

    # user least squares optimization to selectively modulate the
    # source luminance differences

    # intialize gray to be luminance channel
    g = L[:,:]
    print shape(g)
    print shape(sigmas)
    print sum(sigmas)

    for i in range(h*w):
        for j in range(h*w):
            
    
    
    poisson = PoissonCG(sigmas, g, 50)
    poisson = XYZ2RGB(CIELAB2XYZ(poisson))
    
    return poisson


def PoissonCG(bg, fg, niter):
    h, w, q = shape(fg)
    b = Laplacian(fg)
    x = bg
    r = b - Laplacian(x) # r_0
    d = r # d_0
    for i in range(niter):
        a = dotIm(r, r) / dotIm(d, Laplacian(d))
        x += a*d # x_i+1
        r_ = r - a*Laplacian(d) # r_i+1
        B = dotIm(r_, r_) / dotIm(r, r)
        d = r_ + B*d # d_i+1
        r = r_ # r_i+1
    return x


def BW(im):
    bw = constantIm(height(im), width(im), 0.0)
    w = array([0.3, 0.6, 0.1])
    for y, x in imIter(bw):
        bw[y, x] = dot(im[y, x], w)
    return bw

