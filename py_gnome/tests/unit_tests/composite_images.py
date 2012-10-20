#!/usr/bin/env python

"""
composite the foreground on top of the background images
"""

import PIL.Image

back = PIL.Image.open("Test_images/background_map.png")
fore = PIL.Image.open("Test_images/foreground_00000.png")

#back.paste(fore, (0,0))

back = PIL.Image.blend(back, fore, 0.5) 

back.save("Test_images/full_map_00000.png")

