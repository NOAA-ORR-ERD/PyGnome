#Creates a particle png from a particle plot png and a location file png
import Image
from os import * 
import numpy as np
from matplotlib.pyplot import * 

#default for prototype
a = .1
spillstack = [] 
imgstack = []

for i in range(100):
    x = np.random.rand(10000)
    fig = figure(figsize=(5.00,5.00),dpi=100)
    plot(x,'.')
    k = str(i)
    spill = fig.savefig("spill"+k+".png")
    spillstack.append(spill)
    im1 = Image.open("li.png")
    im2 = Image.open("spill"+k+".png")
    im2 = im2.convert("RGB")
    print "printing",im1.mode,im2.mode,im1.size,im2.size
    ppng = lambda alpha: Image.blend(im1,im2,alpha)
    img = ppng(a).save("ppng"+k+".png")
    imgstack.append(img)
    
print imgstack