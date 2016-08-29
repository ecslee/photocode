import os.path, colorsys

def execute(filename, fpath, brightness_correction=False):
    modified_filename = "%s-%s-%s" % ('color2gray', brightness_correction, filename)
    head, tail = os.path.split(fpath)
    # Save transformed image to the cache dir of the page
    #head = head.replace('attachments', 'cache') 
    modified_fpath = os.path.join(head, modified_filename)

    # Look if requested image is already available
    if os.path.isfile(modified_fpath):
        return (modified_filename, modified_fpath)

    helpers_available = True
    try:
        from PIL import Image
        import numpy
    except:
        helpers_available = False
    if not helpers_available:
        return (filename, fpath)

    # Get image data
    im = Image.open(fpath)
    if im.mode in ['1', 'L']: # Don't process black/white or grayscale images
        return (filename, fpath)
    #im = im.copy() 
    im = im.convert('RGB')
    pix = im.load()

    # Color to gray conversion according to Martin Faust
    # copyright: 2005 by martin.faust@e56.de
    # license: GNU GPL
    min = 100.0
    max = -100.0
    mean = 0.0
    im_x, im_y = im.size
    gray = numpy.zeros([im_x, im_y], float)
    for y in range(im_y):
        for x in range(im_x):
            #r, g, b = im.getpixel((x, y))
            r, g, b = pix [x, y]
            #hue, saturation, brightness = rgb2hsl(r, g, b)
            #hue, brightness, saturation = colorsys.rgb_to_hls(r, g, b)
            hue, saturation, brightness = colorsys.rgb_to_hsv(r, g, b)
            hue = hue * 360.0
                
            if saturation == 0.0:
                gray[x][y] = 1.5 * brightness
            else:
                gray[x][y] = brightness +  brightness*saturation
                
            if gray[x][y] < min:
                min = gray[x, y]
            if gray[x][y] > max:
                max = gray[x][y]
            mean += gray[x][y]
    
    mean /= (float) (im_y * im_x)
    min = 0.0
    max = (mean + max) * 0.5

    for y in range(im_y):
        for x in range(im_x):
            if brightness_correction:
                brightness = 0.9 * 255.0 * (gray[x][y] - min) / (max - min)
            else:
                brightness = 255.0 * (gray[x][y] - min) / (max - min)
            if brightness > 255.0:
                brightness = 255.0
            if brightness < 0.0:
                brightness = 0.0
            #im.putpixel((x, y), (int(brightness), int(brightness), int(brightness)))
            pix [x, y] = (int(brightness), int(brightness), int(brightness))

    # Save transformed image
    im.save(modified_fpath)
    return (modified_filename, modified_fpath)


if __name__ == '__main__':
    import sys
    print "Color2Gray image correction for color blind users"
    
    if len(sys.argv) != 3:
        print "Calling syntax: color2gray.py [fullpath to image file] [brightness_correction=True/False]"
        print "Example: color2gray.py C:/wikiinstance/data/pages/PageName/attachments/pic.png False"
        sys.exit(1)

    if not (os.path.isfile(sys.argv[1])):
        print "Given file does not exist"
        sys.exit(1)

    extpos = sys.argv[1].rfind(".")
    if not (extpos > 0 and sys.argv[1][extpos:].lower() in ['.gif', '.jpg', '.jpeg', '.png', '.bmp', '.ico', ]):
        print "Given file is not an image"
        sys.exit(1)

    path, fname = os.path.split(sys.argv[1])
    print "Please wait. Processing image..."

    brightness_correction = bool(sys.argv[2]=='True')

    modified_filename, modified_fpath = execute(fname, sys.argv[1], brightness_correction)
##    import cProfile
##    cProfile.run("execute(fname, sys.argv[1], brightness_correction)")
    if modified_fpath == sys.argv[1]:
        print "Error while processing image: PIL/NumPy missing and/or source file is already a grayscale image."
    else:
        print "Image successfully processed"
