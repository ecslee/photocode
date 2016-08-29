# eseitz - Emily Seitz
# 3/19/12
# 6.815 pset 6

import imageIO
from imageIO import *
import numpy
from numpy import *
import a2
from a2 import *
import time
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

def getBlackPadded(im, y, x):
    if (x<0) or (x>=im.shape[1]) or (y<0) or (y>=im.shape[0]):
        return array([0,0,0])
    else:
        return im[y,x]

# 2 - Warp by a homography
def applyhomography(source, out, H, bilinear=False):
    if not bilinear:
        for y, x in imIter(out):
            pixel = dot(H, [y, x, 1])
            if clipY(source, pixel[0]/pixel[2])==pixel[0]/pixel[2] and clipX(source, pixel[1]/pixel[2])==pixel[1]/pixel[2]:
                out[y, x] = source[pixel[0]/pixel[2]][pixel[1]/pixel[2]]
    else:
        for y, x in imIter(out):
            pixel = dot(H, [y, x, 1])
            if clipY(source, pixel[0]/pixel[2])==pixel[0]/pixel[2] and clipX(source, pixel[1]/pixel[2])==pixel[1]/pixel[2]:
                out[y, x] = interpolateLin(source, pixel[0]/pixel[2], pixel[1]/pixel[2])
    return

def applyhomographyT(source, out, H, T, bilinear=False):
    if not bilinear:
        for y, x in imIter(out):
            pixel = dot(H, [y, x, 1])
            if clipY(source, pixel[0]/pixel[2])==pixel[0]/pixel[2] and clipX(source, pixel[1]/pixel[2])==pixel[1]/pixel[2]:
                out[y+T[0][2], x+T[1][2]] = source[pixel[0]/pixel[2]][pixel[1]/pixel[2]]
            else:
                out[y+t[0][2], x+T[1][2]] = out[y, x]
    else:
        for y, x in imIter(out):
            pixel = dot(H, [y, x, 1])
            #print pixel
            if clipY(source, pixel[0]/pixel[2])==pixel[0]/pixel[2] and clipX(source, pixel[1]/pixel[2])==pixel[1]/pixel[2]:
                out[clipY(out, y+T[0][2]), clipX(out, x+T[1][2])] = interpolateLin(source, pixel[0]/pixel[2], pixel[1]/pixel[2])
            else:
                out[clipY(out, y+T[0][2]), clipX(out, x+T[1][2])] = out[y, x]
    return


# 3 - Solve for a homography from four pairs of points
def computehomography(listOfPairs):
    A = []
    B = []
    for pair in listOfPairs:
        y = pair[0][0]
        x = pair[0][1]
        y_ = pair[1][0]
        x_ = pair[1][1]
        A.append([y, x, 1, 0, 0, 0, -y*y_, -x*y_])#, -y_])
        A.append([0, 0, 0, y, x, 1, -y*x_, -x*x_])#, -x_])
        B.append(y_)
        B.append(x_)
    #A.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
    A_inv = linalg.inv(numpy.array(A))
    #B.append(1)
    B = numpy.array(B)

    # A * x = B
    #     x = A_inv * B
    xVec = dot(A_inv, B)
    xVec.resize(9)
    xVec[8] = 1.0
    return reshape(xVec, (3, 3))


def test(title):
    if title == "poster":
        src = imread("poster.png")
        h, w = src.shape[0]-1, src.shape[1]-1
        pointListPoster = [array([0, 0, 1]), array([0, w, 1]), array([h, w, 1]), array([h, 0, 1])]
        pointListT = [array([170, 95, 1]), array([171, 238, 1]), array([233, 235, 1]), array([239, 94, 1])]
        listOfPairs = zip(pointListT, pointListPoster)
        H = computehomography(listOfPairs)
        out = imread("green.png")
        applyhomography(src, out, H, False)
        imwriteSeq(out)
    elif title == "stata":
        im1 = imread('pano/stata-1.png')
        im2 = imread('pano/stata-2.png')
        pointList1 = [array([209, 218, 1]), array([425, 300, 1]), array([209, 337, 1]), array([396, 336, 1])]
        pointList2 = [array([232, 4, 1]), array([465, 62, 1]), array([247, 125, 1]), array([433, 102, 1])]
        listOfPairs = zip(pointList1, pointList2)
        H = computehomography(listOfPairs)
        out = im1
        applyhomography(im2, out, H, True)
        imwriteSeq(out)
    elif title == "science":
        im1 = imread('pano/science-1.png')
        im2 = imread('pano/science-2.png')
        pointList1 = [array([310, 106, 1]), array([198, 103, 1]), array([168, 74, 1]), array([305, 37, 1])]
        pointList2 = [array([299, 305, 1]), array([185, 292, 1]), array([158, 261, 1]), array([297, 234, 1])]
        listOfPairs = zip(pointList1, pointList2)
        H = computehomography(listOfPairs)
        out = im1*0.2
        applyhomography(im2, out, H, True)
        imwriteSeq(out)
    elif title == "emily":
        out = imread('out15.png')
        src = imread('number_man.png')
        h, w = src.shape[0]-1, src.shape[1]-1
        pointListT = [array([169, 97, 1]), array([171, 234, 1]), array([229, 233, 1]), array([236, 97, 1])]
        pointListPoster = [array([0, 0, 1]), array([0, w, 1]), array([h, w, 1]), array([h, 0, 1])]
        listOfPairs = zip(pointListT, pointListPoster)
        H = computehomography(listOfPairs)
        applyhomography(src, out, H, True)
        imwriteSeq(out)
    return


# 4.1 - Transform a bounding box
def computeTransformBBox(im, H):
    ymin = 0
    ymax = 0
    xmin = 0
    xmax = 0
    for y, x in imIter(im):
            pixel = dot(H, [y, x, 1])
            if pixel[0]/pixel[2] < ymin: ymin = pixel[0]/pixel[2]
            if pixel[0]/pixel[2] > ymax: ymax = pixel[0]/pixel[2]
            if pixel[1]/pixel[2] < xmin: xmin = pixel[1]/pixel[2]
            if pixel[1]/pixel[2] > xmax: xmax = pixel[1]/pixel[2]
    return array([array([ymin, xmin]), array([ymax, xmax])])


# 4.2 - Bounding box union
def bboxUnion(B1, B2):
    return array([array([floor(min(B1[0][0], B2[0][0])), floor(min(B1[0][1], B2[0][1]))]), array([ceil(max(B1[1][0], B2[1][0])), ceil(max(B1[1][1], B2[1][1]))])])


# 4.3 - Translation
def translate(bbox):
    return array([array([1, 0, -bbox[0][0]]), array([0, 1, -bbox[0][1]]), array([0, 0, 1])])


# 4.4 - Putting it all together
def stitch(im1, im2, listOfPairs):
    # (1) compute homography
    H = computehomography(listOfPairs)

    # (2) compute bounding box and translation matrix
    # use im1 as reference
    bbox2 = computeTransformBBox(im2, linalg.inv(H))
    I = array([array([1, 0, 0]), array([0, 1, 0]), array([0, 0, 1])])
    bbox1 = computeTransformBBox(im1, I)
    bbox = bboxUnion(bbox1, bbox2)
    trans = linalg.inv(translate(bbox))

    # (3) composite images
    out = constantIm(bbox[1][0]-bbox[0][0], bbox[1][1]-bbox[0][1], 0.0)
    # first, im1 with black background
    applyhomography(im1, out, trans, False)
    # second, im2 with im1/black
    applyhomography(im2, out, dot(H, trans), True)

    return out


def test2(title):
    t = time.time()
    if title=="stata":
        im1 = imread('pano/stata-1.png')
        im2 = imread('pano/stata-2.png')
        title = "stata_stitch.png"
        pointList1 = [array([209, 218, 1]), array([425, 300, 1]), array([209, 337, 1]), array([396, 336, 1])]
        pointList2 = [array([232, 4, 1]), array([465, 62, 1]), array([247, 125, 1]), array([433, 102, 1])]
    elif title=="emily":
        im2 = imread("sunrise1.png")
        im1 = imread("sunrise2.png")
        title = "sunrise_stitch_RL.png"
        pointList2=[array([121, 303, 1]), array([185, 222, 1]), array([218, 421, 1]), array([244, 276, 1])]
        pointList1=[array([111, 135, 1]), array([176, 45, 1]), array([208, 247, 1]), array([237, 106, 1])]
    elif title=="science":
        im1 = imread('pano/science-1.png')
        im2 = imread('pano/science-2.png')
        title = "science_stitch.png"
        pointList1=[array([182, 132, 1]), array([322, 77, 1]), array([189, 46, 1]), array([386, 116, 1])]
        pointList2=[array([164, 317, 1]), array([313, 273, 1]), array([182, 231, 1]), array([378, 317, 1])]
    elif title=="monu":
        im2 = imread("pano/monu-1.png")
        im1 = imread("pano/monu-2.png")
        title = "monu_stitch_RL.png"
        pointList2=[array([120, 211, 1]), array([157, 417, 1]), array([295, 389, 1]), array([237, 307, 1])]
        pointList1=[array([107, 5, 1]), array([154, 228, 1]), array([285, 199, 1]), array([236, 121, 1])]
    elif title=="sunset":
        im2 = imread("pano/sunset-1.png")
        im1 = imread("pano/sunset-2.png")
        title = "sunset_stitch_RL.png"
        pointList2=[array([229, 228, 1]), array([178, 261, 1]), array([178, 413, 1]), array([215, 463, 1])]
        pointList1=[array([229, 63, 1]), array([177, 100, 1]), array([181, 252, 1]), array([217, 296, 1])]
    listOfPairs = zip(pointList1, pointList2)
    imwrite(stitch(im1, im2, listOfPairs), title)
    print "took ", time.time()-t, " sec"
    

# 4.5 - N images
def stitchN(listOfImages, listOfListOfPairs, refIndex):
    # pair = im_i and im_i+1
    # (1) compute homographies, H[i] = im[i]&im[i-1]
    I = array([array([1, 0, 0]), array([0, 1, 0]), array([0, 0, 1])])
    listOfH = []
    for i in range(len(listOfImages)):
        listOfH.append(0)
    listOfH[refIndex] = I
    # compute homographies for images with indeces before refIndex
    for i in range(1, refIndex+1):
        pairFlip = []
        for p in listOfListOfPairs[refIndex-i]:
            pairFlip.append([p[1], p[0]])
        listOfH[refIndex-i] = dot(computehomography(pairFlip), listOfH[refIndex-i+1])
    # compute homographies for images with indices after refIndex
    for i in range(refIndex, len(listOfImages)-1):
        listOfH[i+1] = dot(computehomography(listOfListOfPairs[i]), listOfH[i])

    # (2) compute bounding box and translation matrix
    bboxU = computeTransformBBox(listOfImages[0], linalg.inv(listOfH[0]))
    for i in range(1, len(listOfImages)):
        bboxU = bboxUnion(bboxU, computeTransformBBox(listOfImages[i], linalg.inv(listOfH[i])))
    trans = linalg.inv(translate(bboxU))

    # (3) composite images
    out = constantIm(bboxU[1][0]-bboxU[0][0], bboxU[1][1]-bboxU[0][1], 0.0)
    for i in range(len(listOfImages)):
        applyhomography(listOfImages[i], out, dot(listOfH[i], trans), True)

    return out

def stitchNCylin(listOfImages, listOfListOfPairs, refIndex):
    # pair = im_i and im_i+1
    # (1) compute homographies, H[i] = im[i]&im[i-1]
    I = array([array([1, 0, 0]), array([0, 1, 0]), array([0, 0, 1])])
    listOfH = []
    for i in range(len(listOfImages)):
        listOfH.append(0)
    listOfH[refIndex] = I
    # compute homographies for images with indeces before refIndex
    for i in range(1, refIndex+1):
        pairFlip = []
        for p in listOfListOfPairs[refIndex-i]:
            pairFlip.append([p[1], p[0]])
        listOfH[refIndex-i] = dot(computehomography(pairFlip), listOfH[refIndex-i+1])
    # compute homographies for images with indices after refIndex
    for i in range(refIndex, len(listOfImages)-1):
        listOfH[i+1] = dot(computehomography(listOfListOfPairs[i]), listOfH[i])

    # (2) compute bounding box and translation matrix
    bboxU = computeTransformBBox(listOfImages[0], linalg.inv(listOfH[0]))
    for i in range(1, len(listOfImages)):
        bboxU = bboxUnion(bboxU, computeTransformBBox(listOfImages[i], linalg.inv(listOfH[i])))
    trans = linalg.inv(translate(bboxU))

    # (3) composite images
    out = constantIm(bboxU[1][0]-bboxU[0][0], bboxU[1][1]-bboxU[0][1], 0.0)
    for i in range(len(listOfImages)):
        applyhomography(listOfImages[i], out, dot(listOfH[i], trans), True)

    return out


def testN(i):
    im1 = imread("pano/guedelon-1.png")
    im2 = imread("pano/guedelon-2.png")
    im3 = imread("pano/guedelon-3.png")
    im4 = imread("pano/guedelon-4.png")
    listOfImages = [im1, im2, im3, im4]
    listOfListOfPairs = []
    pointList1=[array([82, 208, 1]), array([396, 211, 1]), array([340, 310, 1]), array([67, 330, 1])]
    pointList2=[array([65, 55, 1]), array([393, 26, 1]), array([334, 128, 1]), array([86, 172, 1])]
    listOfListOfPairs.append(zip(pointList1, pointList2))
    pointList2=[array([58, 143, 1]), array([172, 259, 1]), array([436, 300, 1]), array([386, 200, 1])]
    pointList3=[array([16, 18, 1]), array([164, 130, 1]), array([417, 147, 1]), array([375, 52, 1])]
    listOfListOfPairs.append(zip(pointList2, pointList3))
    pointList3=[array([94, 129, 1]), array([393, 80, 1]), array([342, 252, 1]), array([248, 196, 1])]
    pointList4=[array([79, 65, 1]), array([390, 4, 1]), array([335, 181, 1]), array([241, 129, 1])]
    listOfListOfPairs.append(zip(pointList3, pointList4))
    imwriteSeq(stitchN(listOfImages, listOfListOfPairs, i))
    
        



