#!/usr/bin/env python
"""
A module that did contain various functions for reading and writing
assorted HAZMAT file formats.

Now it's just BNA format

Other stuff is now maintained in either the ood_utils package and/or libgoods
"""

import os
import numpy as np
from ..geometry import polygons


def GetNextBNAPolygon(f, dtype=np.float64):
    """
    Utility function that returns the next polygon from a BNA file

    returns: (points, poly_type, name, sname) where:
        points:    Nx2numpy array of floats with the points
        poly_type: one of "point", "line", "poly"
        name:      name defined in the BNA
        sname:     secondary name defined in the BNA

    NOTE: It is the BNA standard to duplicate the first and last points.
          In that case, the duplicated last point is removed.

           "holes" in polygons are not supported in this code.
    See:
       http://www.softwright.com/faq/support/boundary_file_bna_format.html

    NOTE: This code doesn't allow extra spaces around the commas in the
          header line.
          If there are no commas allowed in the name, it would be easier to
          simply split on the commas
          (or march through the line looking for the quotes -- regex?)
    """
    while True:  # skip blank lines
        header = f.readline()
        if not header:  # end of file
            return None
        if header.strip():  # found a header
            break
        else:
            continue

    try:
        fields = header.split('"')
        name = fields[1]
        sname = fields[3]
        num_points = int(fields[4].strip()[1:])
    except (ValueError, IndexError):
        raise ValueError('File has incorrect header for BNA format: {0}'
                         .format(header))

    if num_points < 0 or num_points == 2:
        poly_type = 'polyline'
        num_points = abs(num_points)
    elif num_points == 1:
        poly_type = 'point'
    elif num_points > 2:
        poly_type = 'polygon'
    else:
        raise ValueError("polygon {0} does not have a valid number of points"
                       .format(name))

    if True:  # to keep the indentation for now
        points = np.zeros((num_points, 2), dtype)
        for i in range(num_points):
            line = f.readline()
            if not line:
                raise ValueError(f"empty coords line in {header.strip()}. "
                                 "check number of vertices")
            try:
                points[i, :] = [float(j) for j in line.split(',')]
            except ValueError as err:
                raise ValueError(f"incorrect coords in line: {line} "
                                 f"in poly: {header}") from err

    if poly_type == 'polygon':
        # first and last points are the same in BNA,
        # but we don't want the duplicate point.
        if (points[0, 0] == points[-1, 0] and
                points[0, 1] == points[-1, 1]):
            points = points[0:-1]

    return (points, poly_type, name, sname)


# def WriteBNA(filename, polyset):
#     """
#     Writes a BNA file to filename

#     :param filename: A filename to write to.
#     :param polyset: A geometry.polygons.PolygonSet object, with metadata
#                     (poly_type, name, secondary name)

#     (such as returned by ReadBNA)
#     """
#     outfile = open(filename, 'w')

#     for poly in polyset:
#         m = poly.metadata
#         outfile.write('"%s","%s", %i\n' % (m[1], m[2], len(poly)))

#         for point in poly:
#             # point = np.asarray(point)
#             outfile.write('%.8f, %.8f \n' % (point[0], point[1]))


def ReadBNA(filename, polytype="list", dtype=np.float64):
    """
    Read a bna file.

    :param filename: A filename to write to.

    :param polytype: The type of polygon structure to return.
    :type polytype: one of: ("list", "PolygonSet", "BNADataClass")

    :return: returns the BNA data contents
    :rtype: Results are returned as one of:

            :list: A list of tuples (points, poly_type, name, secondary name)
            :PolygonSet: A geometry.polygons.PolygonSet object,
                         with metadata (poly_type, name, secondary name)
            :BNADataClass: A BNAData class object.  This may be broken now!

    The dtype parameter specifies what numpy data type you want the points
    data in -- it defaults to float (C double)
    """
    fd = open(filename, 'r')

    if polytype == 'list':
        Output = []

        while True:
            poly = GetNextBNAPolygon(fd, dtype=dtype)
            if poly is None:
                break
            Output.append(poly)
    elif polytype == 'PolygonSet':
        Output = polygons.PolygonSet(dtype=dtype)

        while True:
            poly = GetNextBNAPolygon(fd)
            if poly is None:
                break
            # fixme: should this be a dict, instead?
            Output.append(poly[0], poly[1:])

    elif polytype == 'BNADataClass':
        raise TypeError('"BNADataClass" is not supported by this version of the code')
    else:
        raise ValueError('polytype must be either "list" '
                         'or "PolygonSet"')

    fd.close()
    return Output
