#!/usr/bin/env python

"""
Script to experiment with contouring LEs
"""


import numpy as np

from gnome.utilities import map_canvas
from gnome.utilities.geometry import BBox


def fake_LEs(num=1000,
             dist='normal',
             **kwargs):
    """
    generate some fake LEs in various distributions

    options are:

    'normal' -- a guassian blob
    """
    if dist == 'normal':
        std = kwargs.get('std', 0.2)
        center = kwargs.get('center', (0.0,0.0))
        print("computing normal distro with std: {}".format(std))
        positions = np.random.multivariate_normal(center, ((std, 0),(0, std)), size=(num,))
        return positions

def plot_LEs(positions, filename='le_plot.png'):
    """
    plot the LEs and save as an image
    """
    # compute the bounding box:
    bb = BBox.fromPoints(positions)
    # bb.scale(1.1)

    canvas = map_canvas.MapCanvas((500, 500),
                                  viewport=bb,
                                  preset_colors='web',
                                  background_color='white',
                                  )


    canvas.draw_points(positions,
                       diameter=1,
                       color='black',
                       shape="round",
                       background=False
                       )
    canvas.save("test.png")



if __name__ == "__main__":
    pos = fake_LEs(num = 10)
    print(pos)
    plot_LEs(pos)

