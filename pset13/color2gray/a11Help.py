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
from a7help import *
import a11
from a11 import *
import time

def ramp(mask):
    r=arange(0, 1.0, 1.0/mask.shape[1])
    tmp=zeros_like(mask[:, :, 0])
    tmp[:]=r
    out=zeros_like(mask)
    out[:, :, 0]=tmp
    out[:, :, 1]=tmp
    out[:, :, 2]=tmp
    return out


lap2D=array([[0, -1.0, 0], [-1.0, 4.0, -1], [0, -1.0, 0]])
    
def Laplacian(im):
    out=empty_like(im)
    for i in xrange(3):
        out[:, :, i]= signal.convolve(im[:, :, i], lap2D, mode='same')
    return out

def PoissonComposite(bg, fg, mask, y, x, CG=True, useLog=True, niter=200):
    h, w=fg.shape[0], fg.shape[1]
    mask[mask>0.5]=1
    mask[mask<0.6]=0.0
    bg2=(bg[y:y+h, x:x+w]).copy()
    out=bg.copy()
    if useLog:
        bg2[bg2==0]=1e-4
        fg[fg==0]=1e-4
        bg3=log(bg2)+3
        fg3=log(fg)+3
    else:
        bg3=bg2
        fg3=fg
    
    if CG: tmp=PoissonCG(bg3, fg3, mask, niter)
    else: tmp=Poisson(bg3, fg3, mask, niter)

    if useLog:
        out[y:y+h, x:x+w]=exp(tmp-3)
    else: out[y:y+h, x:x+w]=tmp
    return out

def test():
    bg=imread('Poisson/waterpool.png')
    fg=imread('Poisson/bear.png')
    mask=imread('Poisson/mask.png')
    #imwrite( naiveComposite(bg, fg, mask, 50, 1), 'naive.png')
    imwrite(PoissonComposite(bg, fg, mask, 50, 10,  CG=False, useLog=False, niter=250), 'aa-poisson-250.png')
    imwrite(PoissonComposite(bg, fg, mask, 50, 10,  CG=False, useLog=True, niter=250), 'aa-poisson-log-250.png')
    imwrite(PoissonComposite(bg, fg, mask, 50, 10,  CG=True, useLog=False, niter=250), 'aa-poisson-CG-250.png')
    imwrite(PoissonComposite(bg, fg, mask, 50, 10,  CG=True, useLog=True, niter=250), 'aa-poisson-CG-log-250.png')

def apple():
    bg = imread('emily/apple.png')
    fg = imread('emily/sign.png')
    mask = imread('emily/sign_mask.png')
    imwrite(PoissonComposite(bg, fg, mask, 45, 100, True, False, 250), 'apple-CG-250.png')
    imwrite(PoissonComposite(bg, fg, mask, 45, 100, True, True, 250), 'apple-CG-log-250.png')

def mm():
    bg = imread('emily/apple-tiny.png')
    fg = imread('emily/mm-tiny.png')
    mask = imread('emily/mm_mask.png')
    imwrite(PoissonComposite(bg, fg, mask, 15, 20, True, False, 250), 'apple-tiny-CG-250.png')

def blink():
    #bg = imread('blink-big-bg.png')
    bg = imread('blink-log2.png')
    fg = imread('blink-big-fg.png')
    #mask = imread('blink-big-mask-one.png')
    mask = imread('blink-big-mask-two.png')
    #imwrite(PoissonComposite(bg, fg, mask, 48, 119, True, True, 100), 'blink-log2.png')
    imwrite(PoissonComposite(bg, fg, mask, 73, 133, True, True, 1), 'blink-log2-2.png')



def testRamp():
    mask=imread('Poisson/mask3.png')
    bg=ramp(mask)
    fg=ones_like(mask)
 
