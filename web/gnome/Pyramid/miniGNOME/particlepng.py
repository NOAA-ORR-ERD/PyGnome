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