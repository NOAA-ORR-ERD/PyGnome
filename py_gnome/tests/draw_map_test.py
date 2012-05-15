#!usr/bin/env python
"""

simple script to check out BNA reading, map drawing.

"""

import gnome
from gnome.utilities.map_canvas import make_map

from hazpy.file_tools import haz_files

make_map("LMiss.bna", "LMiss.png")

# polygons = haz_files.ReadBNA('LMiss.bna', "PolygonSet")

