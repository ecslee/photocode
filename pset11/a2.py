# eseitz - Emily Seitz
# 2/21/12
# 6.815 A2

import numpy
import imageIO
import math
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


################################
# Functions from lecture slides:
################################

def imIter(im):
    for y in xrange(im.shape[0]):
        for x in xrange(im.shape[1]):
            yield y, x

def getBlackPadded(im, y, x):
    if (x<0) or (x>=im.shape[1]) or (y<0) or (y>= im.shape[0]):
        return numpy.array([0, 0, 0])
    else:
        return im[y, x]

def height(im):
    return im.shape[0]

def width(im):
    return im.shape[1]

def clipX(im, x):
    return min(width(im)-1, max(x, 0))

def clipY(im, y):
    return min(height(im)-1, max(y, 0))

def getSafePix(im, y, x):
    return im[clipY(im, y), clipX(im, x)]

def scaleBAD(im, k):
    out = imageIO.constantIm(im.shape[0]*k, im.shape[1]*k, 0)
    for y, x in imIter(im):
        out[k*y, k*x]= im[y, x]
    return out


################################
# Functions by me:
################################

# 3.1 - basic scaling with nearest neighbor
def scaleNN(im, k):
    scaled = imageIO.constantIm(height(im)*k, width(im)*k, 0.0)
    for y, x in imIter(scaled):
        scaled[y, x] = im[y/k, x/k]
    return scaled


# 3.2 - scaling with bilinear interpretation
def scaleLin(im, k, edge):
    scaled = imageIO.constantIm(height(im)*k, width(im)*k, 0.0)
    for y, x in imIter(scaled):
        scaled[y, x] = interpolateLin(im, y/k, x/k) if edge else interpolateLinBlack(im, y/k, x/k)
    return scaled

def is_int(n):
    return numpy.floor(n) == numpy.ceil(n)

def interpolateLin(im, y, x):
    # assuming edge padding
    y = clipY(im, y)
    x = clipX(im, x)

    if is_int(y):
        if is_int(x):
            return im[y][x]
        else:
            x_only = (x-numpy.floor(x))*im[y][numpy.ceil(x)] + (numpy.ceil(x)-x)*im[y][numpy.floor(x)]
            return x_only
    else:
        if is_int(x):
            y_only = (y-numpy.floor(y))*im[numpy.ceil(y)][x] + (numpy.ceil(y)-y)*im[numpy.floor(y)][x]
            return y_only
        else:
            # linear on x for top
            x_top = (x-numpy.floor(x))*im[numpy.floor(y)][numpy.ceil(x)] + (numpy.ceil(x)-x)*im[numpy.floor(y)][numpy.floor(x)]
            # linear on x for bottom
            x_bot = (x-numpy.floor(x))*im[numpy.ceil(y)][numpy.ceil(x)] + (numpy.ceil(x)-x)*im[numpy.ceil(y)][numpy.floor(x)]
            # linear on y for both
            y_both = abs(y-numpy.ceil(y))*x_top + abs(y-numpy.floor(y))*x_bot
            return y_both

def interpolateLinBlack(im, y, x):
    # using black padding
    if is_int(y):
        if is_int(x):
            return getBlackPadded(im, y, x)
        else:
            x_only = (x-numpy.floor(x))*getBlackPadded(im, y, numpy.ceil(x)) + (numpy.ceil(x)-x)*getBlackPadded(im, y, numpy.floor(x))
            return x_only
    else:
        if is_int(x):
            y_only = (y-numpy.floor(y))*getBlackPadded(im, numpy.ceil(y), x) + (numpy.ceil(y)-y)*getBlackPadded(im, numpy.floor(y), x)
            return y_only
        else:
            # linear on x for top
            x_top = (x-numpy.floor(x))*getBlackPadded(im, numpy.floor(y), numpy.ceil(x)) + (numpy.ceil(x)-x)*getBlackPadded(im, numpy.floor(y), numpy.floor(x))
            # linear on x for bottom
            x_bot = (x-numpy.floor(x))*getBlackPadded(im, numpy.ceil(y), numpy.ceil(x)) + (numpy.ceil(x)-x)*getBlackPadded(im, numpy.ceil(y), numpy.floor(x))
            # linear on y for both
            y_both = abs(y-numpy.ceil(y))*x_top + abs(y-numpy.floor(y))*x_bot
            return y_both


# 4.2 - warping according to one pair of segments
# 2D points = numpy.array(x, y)

# calculates magnitude of a vector
def mag(vector):
    return math.sqrt(vector[0]**2 + vector[1]**2)

# calcuates the dot product between two vectors
def dotProd(vector1, vector2):
    return vector1[0]*vector2[0] + vector1[1]*vector2[1]

# calculates perpindicular vector to a segment
def perp(vector):
    return numpy.array([vector[1], -vector[0]], dtype=numpy.float64)

class segment:
    def __init__(self, x1, y1, x2, y2):
        self.p1 = numpy.array([x1, y1], dtype=numpy.float64)
        self.p2 = numpy.array([x2, y2], dtype=numpy.float64)
        self.vector = numpy.array([x2-x1, y2-y1], dtype=numpy.float64)

    def __add__(self, other):
        return segment(self.p1[0]+other.p1[0], self.p1[1]+other.p1[1], self.p2[0]+other.p2[0], self.p2[1]+other.p2[1])
    
    def __sub__(self, other):
        return segment(self.p1[0]-other.p1[0], self.p1[1]-other.p1[1], self.p2[0]-other.p2[0], self.p2[1]-other.p2[1])

    def __mul__(self, k):
        return segment(self.p1[0]*k, self.p1[1]*k, self.p2[0]*k, self.p2[1]*k)

    def transform(self, point, source):
        segX = segment(self.p1[0], self.p1[1], point[0], point[1])
        u = dotProd(segX.vector, self.vector) / dotProd(self.vector, self.vector)
        v = dotProd(segX.vector, perp(self.vector)) / math.sqrt(dotProd(self.vector, self.vector))
        # x_prime = X in source image
        # p_prime = P in source = source.p1
        # q_prime = Q in source = source.p2
        x_prime = source.p1 + u*source.vector + (v*perp(source.vector))/math.sqrt(dotProd(source.vector, source.vector))
        return x_prime

def warpBy1(im, segmentBefore, segmentAfter):
    warped = imageIO.constantIm(height(im), width(im), 0.0)
    for y, x in imIter(warped):
        x_prime = segmentAfter.transform([x, y], segmentBefore)
        warped[y][x] = im[clipX(im, x_prime[1])][clipY(im, x_prime[0])]
    return warped

# 4.3 - warping according to multiple pairs of segments
def shortestDist(point, line):
    perp = perp(line)
    m = perp[1]/perp[0]
    b = point[1] - m*point[0]

# following pseudocode from Beier pg. 37
def warp(im, listSegmentsBefore, listSegmentsAfter, a=10, b=1, p=1):
    warped = imageIO.constantIm(height(im), width(im), 0.0)
    for y, x in imIter(warped):
        DSUM = (0, 0)
        weightsum = 0.0
        for s in range(len(listSegmentsAfter)):
            x_orig = numpy.array([x, y], dtype=numpy.float64)
            x_prime = listSegmentsAfter[s].transform([x, y], listSegmentsBefore[s])
            segX = segment(listSegmentsAfter[s].p1[0], listSegmentsAfter[s].p1[1], x, y)
            disp = x_prime - x_orig

            u = dotProd(segX.vector, listSegmentsAfter[s].vector) / dotProd(listSegmentsAfter[s].vector, listSegmentsAfter[s].vector)
            v = dotProd(segX.vector, perp(listSegmentsAfter[s].vector)) / math.sqrt(dotProd(listSegmentsAfter[s].vector, listSegmentsAfter[s].vector))
            if (0 <= u < 1):
                dist = abs(v)
            elif (u < 0):
                dist = math.sqrt((x - listSegmentsAfter[s].p1[0])**2 + (y - listSegmentsAfter[s].p1[1])**2)
            elif (u >= 1):
                dist = math.sqrt((x - listSegmentsAfter[s].p2[0])**2 + (y - listSegmentsAfter[s].p2[1])**2)

            weight = ((mag(listSegmentsBefore[s].vector)**p) / (a + float(dist)))**b
            DSUM += disp * weight
            weightsum += weight
        x_prime = x_orig + DSUM / weightsum
        warped[y][x] = im[clipY(im, x_prime[1])][clipX(im, x_prime[0])]
    return warped

# 4.4 - morphing
def morph(im1, im2, listSegmentsBefore, listSegmentsAfter, N=1, a=10, b=1, p=1):
    currentSegments = safe_copy(listSegmentsBefore)
    nextSegments = safe_copy(listSegmentsBefore)

    imageIO.imwrite(im1, 'morph000.png')
    i = 0
    j = 0
    total = []
    for n in range(1, N+1):
        j += 1
        currentSegments = safe_copy(nextSegments)
        step1 = listSegmentsBefore*(1-(n/float(N))) + listSegmentsAfter*(n/float(N))
        step2 = listSegmentsBefore*(n/float(N)) + listSegmentsAfter*(1-(n/float(N)))
        a = warp(im1, listSegmentsBefore, step1)
        b = warp(im2, listSegmentsAfter, step2)
        total.append([a, b])
        print "step: ", j
        #imageIO.imwrite(a*(1-(n/float(N))) + b*(n/float(N)), 'morph'+str('%03d' %i)+'.png')

    for n in range(0, N):
        i+=1
        imageIO.imwrite(total[n][0]*(1-((n+1)/float(N+1))) + total[N-n-1][1]*((n+1)/float(N+1)), 'morph'+str('%03d' %i)+'.png')
    i+=1
    imageIO.imwrite(im2, 'morph' + str('%03d' %i) + '.png')
    pass

# numpy.copy() was not working correctly for my segments,
# so I wrote a safer 
def safe_copy(a):
    b = []
    for x in a:
        b.append(segment(x.p1[0], x.p1[1], x.p2[0], x.p2[1]))
    return numpy.array(b)


def f():
    a = imageIO.imread('fredo2.png')
    b = imageIO.imread('werewolf.png')
    segmentsBefore=numpy.array([segment(90, 129, 108, 128), segment(146, 127, 162, 131), segment(137, 169, 135, 128), segment(99, 199, 135, 198), segment(130, 189, 119, 221)])
    segmentsAfter=numpy.array([segment(83, 114, 108, 110), segment(140, 105, 156, 103), segment(136, 131, 127, 104), segment(102, 171, 146, 168), segment(134, 152, 128, 200)])
    morph(a, b, segmentsBefore, segmentsAfter, 2)

def cousins():
    a = imageIO.imread('cooper.png')
    b = imageIO.imread('westin.png')
    segmentsBefore=numpy.array([segment(62, 117, 85, 118), segment(114, 120, 137, 121), segment(86, 141, 114, 143), segment(73, 162, 120, 166), segment(99, 161, 97, 180), segment(95, 205, 41, 144), segment(148, 155, 101, 204)])
    segmentsAfter=numpy.array([segment(60, 126, 87, 127), segment(114, 128, 138, 130), segment(87, 152, 115, 154), segment(77, 175, 120, 177), segment(99, 173, 99, 185), segment(98, 214, 35, 154), segment(149, 160, 104, 214)])
    morph(a, b, segmentsBefore, segmentsAfter, 3)

def class6815():
    emily = imageIO.imread('class-2.png')
    yang = imageIO.imread('class-3.png')
    segmentsBefore2=numpy.array([segment(71, 96, 90, 96), segment(112, 97, 130, 97), segment(102, 118, 101, 95), segment(89, 121, 114, 120), segment(83, 141, 119, 140), segment(102, 138, 102, 147), segment(92, 171, 62, 146), segment(116, 170, 143, 146), segment(57, 127, 55, 79), segment(147, 126, 146, 76), segment(86, 13, 50, 42), segment(115, 10, 150, 39)])
    segmentsAfter2=numpy.array([segment(69, 99, 87, 97), segment(114, 95, 132, 93), segment(102, 122, 100, 95), segment(88, 123, 116, 120), segment(83, 143, 123, 140), segment(103, 139, 103, 154), segment(94, 173, 61, 152), segment(120, 172, 148, 147), segment(55, 135, 52, 82), segment(153, 129, 149, 79), segment(77, 21, 46, 49), segment(111, 17, 152, 42)])
    segmentsBefore=numpy.array([segment(81, 95, 121, 95), segment(71, 96, 90, 97), segment(111, 98, 131, 97), segment(103, 118, 101, 94), segment(89, 120, 116, 121), segment(83, 141, 121, 139), segment(102, 146, 102, 138), segment(94, 172, 62, 146), segment(106, 172, 141, 148), segment(58, 126, 54, 93), segment(148, 124, 147, 92), segment(93, 87, 63, 86), segment(110, 87, 139, 86), segment(91, 10, 51, 33), segment(110, 8, 148, 33)])
    segmentsAfter=numpy.array([segment(80, 96, 122, 92), segment(70, 97, 88, 98), segment(115, 95, 133, 93), segment(101, 121, 101, 95), segment(87, 123, 117, 122), segment(84, 143, 123, 140), segment(103, 153, 102, 140), segment(98, 175, 63, 154), segment(112, 175, 150, 146), segment(55, 135, 51, 98), segment(153, 131, 150, 92), segment(90, 84, 61, 88), segment(110, 81, 138, 84), segment(86, 19, 49, 42), segment(105, 16, 150, 41)])
    morph(emily, yang, segmentsBefore, segmentsAfter, 15)

def zacFRon():
    zac = imageIO.imread('zac2.png')
    ron = imageIO.imread('ron2.png')
    segmentsBefore=numpy.array([segment(32, 70, 41, 68), segment(59, 65, 68, 64), segment(41, 94, 66, 90), segment(53, 90, 55, 96), segment(49, 79, 49, 65), segment(44, 83, 60, 81), segment(43, 111, 29, 94), segment(68, 108, 80, 89), segment(43, 13, 19, 31), segment(51, 12, 84, 28), segment(25, 87, 22, 62), segment(80, 83, 80, 54)])
    segmentsAfter=numpy.array([segment(40, 73, 50, 72), segment(67, 72, 76, 74), segment(45, 97, 69, 97), segment(58, 96, 58, 102), segment(59, 85, 58, 72), segment(50, 87, 67, 88), segment(49, 117, 31, 98), segment(65, 118, 81, 100), segment(52, 18, 25, 35), segment(62, 18, 87, 37), segment(28, 90, 28, 63), segment(83, 92, 85, 66)])
    morph(zac, ron, segmentsBefore, segmentsAfter, 15)

def daniNkev():
    dani = imageIO.imread('dani.png')
    kev = imageIO.imread('kevin.png')
    segmentsBefore=numpy.array([segment(105, 124, 130, 128), segment(163, 135, 188, 140), segment(120, 121, 118, 129), segment(178, 134, 176, 141), segment(140, 166, 148, 131), segment(124, 160, 157, 168), segment(111, 179, 161, 190), segment(136, 183, 129, 217), segment(116, 237, 133, 242), segment(109, 228, 84, 170), segment(143, 238, 195, 185), segment(166, 5, 97, 28), segment(175, 7, 228, 48), segment(94, 113, 135, 121), segment(161, 125, 199, 131), segment(82, 157, 94, 86), segment(199, 174, 212, 109)])
    segmentsAfter=numpy.array([segment(101, 116, 127, 120), segment(117, 111, 114, 120), segment(156, 122, 181, 125), segment(170, 117, 168, 126), segment(132, 146, 139, 117), segment(118, 154, 149, 155), segment(105, 180, 161, 187), segment(133, 176, 129, 215), segment(119, 241, 132, 243), segment(112, 234, 80, 175), segment(140, 239, 190, 186), segment(142, 29, 87, 59), segment(150, 28, 216, 85), segment(91, 103, 134, 107), segment(151, 109, 195, 118), segment(78, 165, 86, 92), segment(192, 178, 209, 96)])
    morph(dani, kev, segmentsBefore, segmentsAfter, 15)
    
    
