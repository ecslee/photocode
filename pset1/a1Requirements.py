import a1
import imageIO
import numpy
import random
import unittest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.im = imageIO.imread()

    def test_0_imageLoad(self):
        self.assertEqual(self.im.shape, (85, 128, 3), "Size of input image is wrong. Have you modified in.png by accident?")

    def test_1_brightness(self):
        bright = a1.brightness(self.im,1.5)
        comparison = numpy.array([ 0.55288638, 0.60681123, 0.53061135, 0.21297044, 0.20414782])
        self.assertTrue((numpy.abs((bright[40:45,60,2]/comparison)-1) < .0001).all())

    def test_2_contrast(self):
        contrasted = a1.contrast(self.im, 0.5, 0.6)
        comparison = numpy.array([ 0.30146616, 0.30387551, 0.39698608, 0.63652454, 0.36951123])
        self.assertTrue((numpy.abs((contrasted[45,30:35,1]/comparison)-1) < .0001).all())

    def test_3_frame(self):
        framed = a1.frame(self.im)
        self.assertEqual(framed[int(85*random.random()),-1,int(3*random.random())],0)
        self.assertEqual(framed[-1,int(128*random.random()),int(3*random.random())],0)

    def test_4_bw(self):
        bw = a1.BW(self.im, [0.5, 0.2, 0.3])
        comparison = numpy.array([ 0.33086951, 0.29580893, 0.300243, 0.3047134, 0.31376337])
        self.assertTrue((numpy.abs((bw[25,55:60,0]/comparison)-1) < .0001).all())

    def test_5_brightnessContrastLumi(self):
        bcl = a1.brightnessContrastLumi(self.im, 0.6, 1.5, 0.3)
        self.assertTrue((numpy.abs((self.im[70,80:85,0]/imageIO.imread()[70,80:85,0])-1) < .0001).all())
        comparison = numpy.array([ 0.29026288, 0.30050848, 0.30629434, 0.33570896, 0.31796292])
        self.assertTrue((numpy.abs((bcl[70,80:85,0]/comparison)-1) < .0001).all())

    def test_6_saturate(self):
        saturated = a1.saturate(self.im,2)
        comparison = numpy.array([ 0.41340925, 0.47600393, 0.2669519, 0.02544514, 0.19110046])
        self.assertTrue((numpy.abs((saturated[20:25,115,2]/comparison)-1) < .0001).all())

    def test_7_spanishcastle(self):
        a1.spanish(self.im)
        self.assertTrue(True)

    def test_8_histogram(self):
        histogram = a1.histogram(self.im,20)
        comparison = numpy.array([2.,2.,5.,42.,29.,13.,54.,39.,91.,19.])
        self.assertTrue((numpy.abs((numpy.floor(histogram[0:10]*300)/comparison)-1) < .0001).all())

    def test_9_printhisto(self):
        a1.printHisto(self.im,20,300)
        self.assertTrue(True)

    def test_extracredit_brightnessclipping(self):
        superbright = a1.brightness(self.im, 10)
        self.assertEqual(superbright[29,47,int(3*random.random())],1.0)

    def test_extracredit_contrastclipping(self):
        supersharp = a1.contrast(self.im, 10)
        self.assertEqual(supersharp[41,50,int(3*random.random())],1.0)
        self.assertEqual(supersharp[42,50,int(3*random.random())],0.0)

suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)
