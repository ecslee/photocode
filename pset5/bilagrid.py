import numpy
from numpy import *
import scipy
from scipy import ndimage
import imageIO
from imageIO import *

def height(im):
    return im.shape[0]

def width(im):
    return im.shape[1]


def imIter(im, debug=False, lim=1e6):
    for y in xrange(min(height(im), lim)):
        if debug & (y%10==0): print 'y=', y
        for x in xrange(min(lim, width(im))): yield y, x


class bilaGrid:
      def __init__(self, ref, sigmaS, sigmaR, factor=3.0):
            self.sigmaS=sigmaS
            self.sigmaR=sigmaR
            self.fxy=1.0/sigmaS*factor
            self.fr=1.0/sigmaR*factor
            self.factor=factor
            self.height=ref.shape[0]*self.fxy+2
            self.width=ref.shape[1]*self.fxy+2
            self.miniR=min(ref.flatten())
            self.offsetr=-self.miniR*self.fr+0.5
            self.range=(max(ref.flatten())-self.miniR)*self.fr+2
            self.grid=zeros([self.height, self.width, self.range, 3])
            self.weight=zeros([self.height, self.width, self.range])
            self.ref=ref
            
      def write(self, path='grid'):
            maxi=max(self.grid.flatten())-self.miniR
            for i in xrange(int(self.range)):
                  imwrite((self.grid[:, :, i, :]-self.miniR)/maxi, path+str(i)+'.png')
                          
      def creation(self, im):
            for y, x, in imIter(im):
                  self.grid[y*self.fxy+0.5, x*self.fxy+0.5, self.ref[y, x]*self.fr+self.offsetr]+=im[y, x]
                  self.weight[y*self.fxy+0.5, x*self.fxy+0.5, self.ref[y, x]*self.fr+self.offsetr]+=1.0
            self.im=im
            
      def blur(self):
            self.grid=ndimage.filters.gaussian_filter(self.grid, [self.factor, self.factor, self.factor, 0])
            self.weight=ndimage.filters.gaussian_filter(self.weight, [self.factor, self.factor, self.factor])
            
      def slicing(self):
            h=self.ref.shape[0]
            w=self.ref.shape[1]
            newy, newx=self.fxy*mgrid[0:h, 0:w]
            coord=array([newy, newx, self.ref*self.fr+self.offsetr])
            out=empty([h, w, 3])
            weight=ndimage.map_coordinates(self.weight, coord, order=1)+0.0000000001
            for i in xrange(3):
                  out[:, :, i]=ndimage.map_coordinates(self.grid[:, :, :, i], coord, order=1)/weight
            return out
                  
      def slicingMathieu(self):
            h=self.ref.shape[0]
            w=self.ref.shape[1]
            newy, newx=self.fxy*mgrid[0:h, 0:w]
            coord=array([newy, newx, self.ref*self.fr+self.offsetr])
            out=empty([h, w, 3])
            #self.write('grid-')
            weights=empty([h, w, 3])      
            weight=ndimage.map_coordinates(self.weight, coord, order=1)+0.0000000001
            for i in xrange(3):
                  out[:, :, i]=ndimage.map_coordinates(self.grid[:, :, :, i], coord, order=1)/weight
                  weights[:, :, i]=weight
            maxw=max(weight.flatten())
            weights/=maxw
            #imwrite(weights, 'weights-bila.png')
            #imwrite(abs(self.im), 'self.im.png')
            return out*weights+(1.0-weights)*self.im

      def doit(self, im):
            self.creation(im)
            self.blur()
            return self.slicingMathieu()

