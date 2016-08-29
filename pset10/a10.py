# eseitz - Emily Seitz
# 4/23/12
# 6.815 pset 10

import imageIO
from imageIO import *
import numpy
from numpy import *
import scipy.ndimage
import random
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


# 2.1 - Light field loader
def loadLF(path='chess/chess-tiny-'):
    numFiles = 0
    pics = []
    while True:
        try:
            thisPath = "%(path)s%(#)03d.png" %{"path": path, "#": numFiles}
            imread(thisPath)
            pics.append(thisPath)
            numFiles += 1
        except IOError:
            break
    N = int(sqrt(numFiles))

    h, w = shape(imread(pics[0]))[0:2]
    LF = zeros([N, N, h, w, 3])
    for v in range(N):
        for u in range(N):
            LF[v, u] = imread(pics[N*v+u])
    return LF


# 2.2 - Visualizations
def apertureView(LF):
    N = len(LF)
    h, w = shape(LF[0][0])[0:2]
    print "dimensions: ", N*h, " x ", N*w
    visualize = constantIm((N+1)*h, (N+1)*w, [0.1, 0.0, 0.0])
    orig = LF[0][0]
    for y, x in imIter(orig):
        for v in range(N):
            for u in range(N):
                visualize[(N+1)*y+v, (N+1)*x+u] = LF[v, u, y, x]
    return visualize

def epiSlice(LF, y):
    N = len(LF)
    epislice = LF[N/2,:,y,:,:]
    return epislice


# 2.3 - Image shifting
def ytranslateIm(im, dy, dx):
    h, w = shape(im)[0:2]
    I = matrix([[1, 0], [0, 1]])
    translate = constantIm(height(im), width(im), 0.0)
    for c in range(3):
        translate[:, :, c] = scipy.ndimage.interpolation.affine_transform(im[:,:,c], I, (dy, dx))
    return translate


# 2.4 - Refocusing
def refocusLF(LF, maxParallax=0.0, aperture=17):
    refocused = constantIm(height(LF[0][0]), width(LF[0][0]), 0.0)
    aperture = min(aperture, len(LF))
    a_lo = max(0, (len(LF)-aperture)/2)
    a_hi = a_lo+aperture
    
    for v in range(a_lo, a_hi):
        for u in range(a_lo, a_hi):
            dy = (v-aperture/2)/float(aperture/2) * maxParallax
            dx = (u-aperture/2)/float(aperture/2) * maxParallax
            refocused[:,:] += ytranslateIm(LF[v, u], dy, -dx)
    return refocused/float(aperture**2)


# 2.5 - Rack focus
def rackFocus(LF, minmaxPara=-7.0, maxmaxPara=2.0, step=0.5, aperture=17):
    for p in arange(minmaxPara, maxmaxPara+step, step):
        imwriteSeq(refocusLF(LF, p, aperture), "rackFocus-")
    return


# 3.1 - Sharpness evaluation
def sharpnessMap(im, exponent=1, sigma=1):
    # work in luminance domain
    lumi = dot(im, array([0.3, 0.6, 0.1]))
    lumi = clip(lumi, 1e-6, 1.0)
    # compute high pass by sibtracting a Gaussian blur with std sigma
    hiPass = lumi - scipy.ndimage.filters.gaussian_filter(lumi, sigma)
    # square hi pass and apply Gaussian blur of std 4*sigma
    hiFreq = scipy.ndimage.filters.gaussian_filter(hiPass**2, 4*sigma)
    # raise the result of the second Gaussian blur to the exponent
    hiFreq = hiFreq**exponent
    # normalize
    hiFreq /= hiFreq.max()
    hiFreqIm = constantIm(height(im), width(im), 0.0)
    hiFreqIm[:,:,0] = hiFreq[:,:]
    hiFreqIm[:,:,1] = hiFreq[:,:]
    hiFreqIm[:,:,2] = hiFreq[:,:]
    return hiFreqIm


# 3.2 - Compositing
def fullFocusLinear(imL):
    fullFocus = constantIm(height(imL[0]), width(imL[0]), 0.0)
    sharp = constantIm(height(imL[0]), width(imL[0]), 0.0)
    for im in imL:
        sharpness = sharpnessMap(im, 4)
        imwriteSeq(sharpness)
        sharp += sharpness
        fullFocus += im*sharpness
    return fullFocus/sharp


# 3.3 - Index visualization
def indexVis(imL):
    fullFocus = constantIm(height(imL[0]), width(imL[0]), 0.0)
    sharp = constantIm(height(imL[0]), width(imL[0]), 0.0)
    for im in imL:
        color = [.1*random.randint(0,10), .1*random.randint(0,10), .1*random.randint(0,10)]
        sharpness = sharpnessMap(im)
        sharp += sharpness
        fullFocus += sharpness*color
    return fullFocus/sharp


# 3.4 - Putting it all together
def legoFocus():
    #LF = loadLF('lego/lego-small-full-')
    #rackFocus(LF, -7, 19, 0.5)
    #numFiles = 79
    pics = []
    path = 'legoFocus/rackFocus-'
    for numFiles in range(90, 126):
        try:
            thisPath = "%(path)s%(#)d.png" %{"path": path, "#": numFiles}
            imread(thisPath)
            pics.append(thisPath)
            numFiles += 1
        except IOError:
            break
        numFiles += 1
    for n in range(len(pics)):
        pics[n] = imread(pics[n])
    imwriteSeq(fullFocusLinear(pics))
    imwriteSeq(indexVis(pics))


def testFullFocus(path):
    numFiles = 0
    pics = []
    while True:
        try:
            thisPath = "%(path)s%(#)03d.png" %{"path": path, "#": numFiles}
            imread(thisPath)
            pics.append(thisPath)
            numFiles += 1
        except IOError:
            break
    for n in range(len(pics)):
        pics[n] = imread(pics[n])
    imwriteSeq(fullFocusLinear(pics))
    imwriteSeq(indexVis(pics))
    

def test(name):
    if name=="lego":
        LF = loadLF('lego/lego-small-denser-')
        v = apertureView(LF)
        imwriteSeq(v, 'lego-small-aperature-')
        #e = epiSlice(LF, 100)
        #imwriteSeq(e, 'lego-small-epislice-')
        #r = refocusLF(LF, 5.0, 17)
        #imwriteSeq(r, 'lego-small-refocus-')
        #rackFocus(LF)
        #rackFocus(LF, -7.0, 2.0, 0.5, 5)
    elif name=='chess':
        LF = loadLF('chess/chess-small-full-')
        v = apertureView(LF)
        #imwriteSeq(v, 'chess-small-aperature-')
        e = epiSlice(LF, 100)
        #imwriteSeq(e, 'chess-small-epislice-')
        r = refocusLF(LF, 5.0, 17)
        imwriteSeq(r, 'chess-small-refocus-')
        rackFocus(LF)
    elif name=='test':
        LF = loadLF('chess/chess-tiny-')
        v = apertureView(LF)
        imwriteSeq(v, 'chess-tiny-aperature-')
        #e = epiSlice(LF, 100)
        #imwriteSeq(e, 'chess-tiny-epislice-')
        #r = refocusLF(LF, 5.0, 17)
        #imwriteSeq(r, 'chess-tiny-refocus-')
        #rackFocus(LF)
    return


