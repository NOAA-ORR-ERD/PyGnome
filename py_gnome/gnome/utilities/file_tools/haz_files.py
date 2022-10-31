#!/usr/bin/env python
"""
A module that contains various functions for reading and writing
assorted HAZMAT file formats.

Some of these require the point in polygon code that is in the TAP
check_receptors extension module. I should put that in another library.
"""

import os
import numpy as np

# try:
#     from .filescanner import scan
#     FILESCANNER = True
# except:
FILESCANNER = False #because py3

## fixme: It would be MUCH cleaner to internally store VerDat data with
## Python style slicing and indexing, including storing a 0 at the beginning
##
## It would also be good to re-factor so that the attributes (depths,
## etc) generally are stored in the same array as the point coordinates.


class FileToolsException(Exception):
    '''
        The base class for all exceptions in the FileTools module
    '''
    pass


class BnaError(FileToolsException):
    pass


class BNAData:
    '''
        Class to store the full set of data in a BNA file
    '''
    ##fixme: This needs methods to add polygons one by one
    def __init__(self, PointsData=None, Names=None, Types=None, Filename=None):
        '''
            :param PointsData: A sequence of numpy Nx2 arrays
                               of the points (x,y)
                               i.e. long,lat
            :param Names: A sequence of stings for the names of the polygons
            :param Types: A sequence of strings for the types of the polygons.
        '''
        self.PointsData = PointsData
        self.Filename = Filename
        self.Names = Names
        self.Types = Types

        try:
            l1 = len(PointsData)
        except TypeError:
            l1 = 0

        try:
            l2 = len(Names)
        except TypeError:
            l2 = 0

        try:
            l3 = len(Types)
        except TypeError:
            l3 = 0

        if l1 != l2 != l3:
            raise TypeError('PointsData, Types, and Names must be '
                            'the same length')

    def __getitem__(self, index):
        return (self.PointsData[index], self.Names[index])

    def __len__(self):
        return len(self.PointsData)

    def __str__(self):
        return 'BNAData instance: {0} polygons'.format(len(self))

    def Save(self, filename=None):
        if not filename:
            filename = self.filename

        fd = open(filename, 'w')
        for i, points in enumerate(self.PointsData):
            fd.write('"%s","%s", %i\n' % (self.Names[i],
                                            self.Types[i],
                                            len(points)))
            for p in points:
                fd.write("%.12f,%.12f\n" % (tuple(p)))


def ReadDOGSFile(filename):

    # Read in the DOGS data:
    fd = open(filename, 'r')
    Header = {}

    while 1:
        line = fd.readline().strip()
        if not line[0] == '[':  # done with header
            break
        line = line.split(']')
        # build dictionary of header data
        Header[line[0][1:]] = ''.join(line[1]).strip()

        # initialize arrays

    Npoints = int(Header['NOBJ'])
    Coords = np.zeros((Npoints, 2), dtype=np.float64)
    Depths = np.zeros((Npoints,), dtype=np.float64)

    line = line.split(',')
    for n in range(Npoints):
        lon, lat, depth = (float(val) for val in line[1:4])
        Coords[n, :] = (lon, lat)
        Depths[n] = depth
        line = fd.readline().strip().split(',')

    return (Coords, Depths, Header)


def WriteDOGSFiles(filename, Coords, Depths, Units='meters'):
    N = len(Coords)

    # compute bounding box:
    low_long, low_lat = min(Coords)
    high_long, high_lat = max(Coords)

    fd = open(filename, 'w')

    # I suppose this might change, but then this function will have to also.
    fd.write("[VERS]  2.00\n")
    fd.write("[NOBJ]  %i\n" % N)
    fd.write("[UNIT]  %s\n" % Units)
    fd.write("[BNDS]  %f, %f, %f, %f\n" % (high_lat, low_long,
                                           low_lat, high_long))

    for i in range(N):
        fd.write("%i, %f, %f, %f, 0.0, 0.0, 0\n" % (i + 1,
                                                    Coords[i, 0],
                                                    Coords[i, 1],
                                                    Depths[i]))


def ReadVerdatFile(filename):
    """
    This function reads a DOGS style verdat file

    The output is zero-index based, rather than 1-index based as it is
    in the file.

    ## fixme -- this doesn't keep the units if they are there in the header
    """
    infile = open(filename, 'r')
    PointData = []

    while 1:
        line = infile.readline().strip().split(',')
        if line[0][:4] == 'DOGS':
            continue

        data = tuple(map(float, line))
        if data == (0, 0, 0, 0):
            break

        PointData.append(data[1:])

    NumBoundaries = int(infile.readline())
    Boundaries = np.zeros((NumBoundaries,), np.int32)
    for i in range(NumBoundaries):
        PointNum = int(infile.readline()) - 1  # correcting to be zero-indexed
        Boundaries[i] = PointNum
    infile.close()

    return (np.array(PointData), Boundaries)


def WriteVerdatFile(filename, PointData, Boundaries):
    """
    This function writes a verdata file, of the "DOGS" type.

    The data passed in must be Verdat legal, and is zero-index based.
    """
    fd = open(filename, 'w')

    fd.write('DOGS\n')
    for i in range(len(PointData)):
        fd.write("%i, " % (i + 1))  # Verdat indexes from 1
        fd.write("%f, %f, %f\n" % tuple(PointData[i]))
    fd.write("  0,   0.000,   0.000,   0.000\n")
    fd.write("%i\n" % len(Boundaries))
    for b in Boundaries:
        fd.write("%i\n" % (b + 1))  # Verdata indexes from 1

    fd.close()


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

    # try:
    #     header.decode('ascii')
    # except UnicodeDecodeError:
    #     raise ValueError('File has incorrect header for BNA format')

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
        raise BnaError("polygon {0} does not have a valid number of points"
                       .format(name))

    if FILESCANNER:
        points = scan(f, num_points * 2)
        points = np.asarray(points, dtype=dtype)
        points.shape = (-1, 2)
    else:
        points = np.zeros((num_points, 2), dtype)
        for i in range(num_points):
            line = f.readline()
            if not line:
                raise ValueError(f"empty coords line in {header.strip()}. check number of vertices")
            try:
                points[i,:] = [float(j) for j in line.split(',')]
            except ValueError as err:
                raise ValueError(f"incorrect coords in line: {line} in poly: {header}") from err

    if poly_type == 'polygon':  # first and last points are the same in BNA,
                                # but we don't want the duplicate point.
        if (points[0, 0] == points[-1, 0] and
            points[0, 1] == points[-1, 1]):
            points = points[0:-1]

    return (points, poly_type, name, sname)


def WriteBNA(filename, polyset):
    """
    Writes a BNA file to filename

    polyset must be a A geometry.polygons.PolygonSet object,
              with metadata- (poly_type, name, secondary name)
    (such as returned by ReadBNA)
    """
    outfile = open(filename, 'w')

    for poly in polyset:
        m = poly.metadata
        outfile.write('"%s","%s", %i\n' % (m[1], m[2], len(poly)))

        for point in poly:
            #point = np.asarray(point)
            outfile.write('%.8f, %.8f \n' % (point[0], point[1]))


def ReadBNA(filename, polytype="list", dtype=np.float64):
    """
    Read a bna file.

    Results are returned as one of:
    - "list": A list of tuples:
              (points, poly_type, name, secondary name)

    - "PolygonSet": A geometry.polygons.PolygonSet object,
                    with metadata- (poly_type, name, secondary name)

    - "BNADataClass": A BNAData class object -- this may be broken now!

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
        from ..geometry import polygons
        Output = polygons.PolygonSet(dtype=dtype)

        while True:
            poly = GetNextBNAPolygon(fd)
            if poly is None:
                break
            # fixme: should this be a dict, instead?
            Output.append(poly[0], poly[1:])

    elif polytype == 'BNADataClass':
        from ..geometry import polygons
        polys = polygons.PolygonSet()
        Types = []
        Names = []
        while 1:
            line = fd.readline()
            if not line:
                break

            line = line.strip()
            Name, line = line.split('","')
            Name = Name[1:]
            Type, line = line.split('",')
            num_points = int(line)
            Types.append(Type)
            Names.append(Name)
            polygon = np.zeros((num_points, 2), np.float64)

            for i in range(num_points):
                polygon[i, :] = (float(val) for val in fd.readline().split(','))
            polys.append(polygon)

        Output = BNAData(polys, Names, Types, os.path.abspath(filename))
    else:
        raise ValueError('polytype must be either "BNADataClass", "list" '
                         'or "PolygonSet"')

    fd.close()
    return Output
