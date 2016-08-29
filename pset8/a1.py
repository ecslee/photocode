# eseitz - Emily Seitz
# 6.815 A1

import imageIO
import numpy

# 3.1a
# changes brightness by multiplying each pixel value by a factor
# clips the brightened array between 0.0 and 1.0
def brightness(im, factor):
    return numpy.clip(factor*im, 0.0, 1.0)


# 3.1b
# f(midpoint) = midpoint
# midpoint = m * midpoint + b
# midpoint = factor * midpoint + b
# clips the brightened array between 0.0 and 1.0
def contrast(im, factor, midpoint=0.3):
    b = midpoint - factor*midpoint
    return numpy.clip(factor*im + b, 0.0, 1.0)


# 3.2
# adds a black border to the image
def frame(im):
    (height, width, x) = numpy.shape(im)
    # left side
    im[:, 0] = 0
    # right side
    im[:, width-1] = 0
    # top
    im[0, :] = 0
    # bottom
    im[height-1, :] = 0
    return im


# 4.1
# converts an image to black and white by taking the dot product of each pixel
# value with an array of weights
def BW(im, weights=[0.3, 0.6, 0.1]):
    imBW = im.copy()
    for row in range(len(imBW)):
        for pixel in range(len(imBW[row])):
            imBW[row][pixel] = numpy.dot(imBW[row][pixel], weights)
    return imBW


# 4.2a
# luminance = BW(im)
# im = luminance * chrominance
# chromi = im / lumi
def lumiChromi(im):
    lumi = BW(im)
    chromi = im / lumi
    return (lumi, chromi)


# 4.2b
# decompose the image into luminance and chrominance,
# process the luminance,
# recombine the luminance and chrominance
def brightnessContrastLumi(im, brightF, contrastF, midpoint=0.3):
    (lumi, chromi) = lumiChromi(im)
    lumi = brightness(lumi, brightF)
    lumi = contrast(lumi, contrastF)
    return lumi * chromi


# 4.3a
# convert RGB to YUV by multiplying by the given matrix
def rgb2yuv(im):
    imYUV = im.copy()
    convert = numpy.matrix('0.299 0.587 0.114; -0.14713 -0.28886 0.436; 0.615 -0.51499 -0.10001')
    for row in range(len(imYUV)):
        for pixel in range(len(imYUV[row])):
            imYUV[row][pixel] = numpy.dot(convert, imYUV[row][pixel])
    return imYUV


# 4.3b
# convert YUV to RGB by multiplying by the given matrix
def yuv2rgb(im):
    imRGB = im.copy()
    convert = numpy.matrix('1 0 1.13983; 1 -0.39465 -0.58060; 1 2.03211 0')
    for row in range(len(imRGB)):
        for pixel in range(len(imRGB[row])):
            imRGB[row][pixel] = numpy.dot(convert, imRGB[row][pixel])
    return imRGB


# 4.3c
# convert RGB to YUV,
# process the pixels in the YUV,
# revert to RGB
def saturate(im, k):
    imYUV = rgb2yuv(im)
    for row in range(len(imYUV)):
        for pixel in range(len(imYUV[row])):
            imYUV[row][pixel] = [1, k, k] * imYUV[row][pixel]
    return yuv2rgb(imYUV)


# 5
# get lumi and chromi aspects,
# convert chromi to YUV,
# process chromi pixels
# revert chromi to RGB
# add a black spot to the center of both pictures
def spanish(im):
    (lumi, chromi) = lumiChromi(im)
    (height, width, x) = numpy.shape(im)

    lumi[height/2, width/2] = 0
    imageIO.imwrite(lumi, 'L.png')
    
    chromi = rgb2yuv(im)
    for row in range(len(chromi)):
        for pixel in range(len(chromi[row])):
            chromi[row][pixel] = [0.5,
                                  -chromi[row][pixel][1],
                                  -chromi[row][pixel][2]]
    chromi[height/2, width/2] = 0
    imageIO.imwrite(yuv2rgb(chromi), 'C.png')


# 6a
# tally pixels in a given range,
# normalize
def histogram(im, N):
    (lumi, chromi) = lumiChromi(im)
    h = numpy.zeros(N)
    for row in range(len(im)):
        for pixel in range(len(im[row])):
            for x in range(0, N+1):
                if (x/float(N) < lumi[row][pixel][0]) and (lumi[row][pixel][0] <= (x+1)/float(N)):
                    h[x] += 1
    return h/sum(h)


# 6b
# print histogram
def printHisto(im, N, scale):
    h = histogram(im, N)
    for b in range(N):
        for x in range(int(numpy.floor(h[b]*scale))):
            print 'X',
        print ''
    return
