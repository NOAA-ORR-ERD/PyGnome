#!/usr/bin/env python

"""
script to compose the GNOME ouput files, and make an animated GIF out of them

this version uses image magick -- we should probably make a pure python one...

"""

import sys
import os

try:
    images_dir = sys.argv[1]
except IndexError:
    images_dir = './'

filelist = os.listdir(images_dir)

background_file = [os.path.join(images_dir, name) for name in filelist
                   if 'background' in name][0]
foreground_files = [os.path.join(images_dir, name) for name in filelist
                    if 'foreground' in name]
foreground_files.sort()

# composit the files:

for name in foreground_files:
    comp_file = os.path.join(images_dir, 'composite' + name[-10:])
    cmd = 'composite -compose atop %s %s %s' % (name, background_file,
            comp_file)
    print cmd
    os.system(cmd)

# build the animated GIF

cmd = 'convert  -delay 20  -loop 0 %s*.png  %s' \
    % (os.path.join(images_dir, 'composite'), os.path.join(images_dir,
       '00-gnome_movie.gif'))
print 'building the GIF:'
print cmd
os.system(cmd)

