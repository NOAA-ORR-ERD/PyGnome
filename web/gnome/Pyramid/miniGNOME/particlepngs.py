#Creates a particle png from a particle plot png and a location file png
import Image
from os import * 
import numpy as np 

#default for prototype
a = .1
im1 = Image.open("li.png")
im2 = Image.open("spill.png")
ppng = lambda alpha: Image.blend(im1,im2,alpha)
ppng(a).save("ppng.png")


#if (im1.size = im2.size)
    #gnomeppng()
#else (im2.resize(im1.size))
    #gnomeppng
print im3.format, im3.size, im3.mode

#called on server to bring in JSON set spill query data
def gnomeppng():
size = 500, 500
for infile in sys.argv[1:]:
    outfile = os.path.splitext(infile)[0] + "ppng" + ".png"
    if infile != outfile:
        try:
            im = Image.open(infile)
            im.thumbnail(size, Image.ANTIALIAS)
            im.save(outfile, "PNG")
        except IOError:
            print "cannot create image for '%s'" % infil