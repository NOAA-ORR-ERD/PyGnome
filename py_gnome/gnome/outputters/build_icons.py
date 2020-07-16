#!/usr/bin/env python
"""
generates a text file with the base64encoded contentes of the icons
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import *
import base64
import glob

icon_files = glob.glob("*.png")

with open('icons.b64', 'w') as outfile:

    for png in icon_files:
        icon_name = png[:-4].upper()
        data = file(png, 'r').read()
        data = base64.b64encode(data)
        outfile.write(icon_name + ' = "')
        outfile.write(data)
        outfile.write('"\n')
