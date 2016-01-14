"""
Polygon module, part of the geometry package

Assorted stuff for working with polygons
"""

import copy

import numpy as np

import BBox


class Polygon(np.ndarray):
    """
    A Polygon class

    This is a subclass of np.ndarray, so that it can be used in place of a
    simple array of points, but also can hold extra meta-data in a "metadata"
    dict.

    """
    def __new__(Polygon, points, metadata=None, copy=True, dtype=np.float):
        # fixme: this needs a better way to index and loop to get a point
        """
        Takes Points as an array. Data is any python sequence that can be
        turned into a Nx2 numpy array of floats. The data will be copied unless
        the copy argument is set to False.

        metadata is a dict of meta-data. This can hold anything.

        """
        # convert to array, copying data unless not requested.
        arr = np.array(points, dtype, copy=copy)
        arr.shape = (-1, 2)     # assure it's the right shape
        # Transform to a Polygon
        arr = arr.view(Polygon)
        # add the attribute
        # Use the specified 'metadata' parameter if given
        if metadata is not None:
            arr.metadata = metadata
        # Otherwise, use points metadata attribute if it exists
        else:
            arr.metadata = getattr(points, 'metadata', {})

        return arr

    def __array_finalize__(self, obj):
        '''
            ndarray subclass instances can come about in three ways:

            - explicit constructor call. This will call the usual sequence
              of SubClass.__new__ then (if it exists) SubClass.__init__.
            - View casting (e.g arr.view(SubClass))
            - Creating new from template (e.g. arr[:3])

            SubClass.__array_finalize__ gets called for all three methods
            of object creation, so this is where our object creation
            housekeeping usually goes.

            I got this from:
              http://www.scipy.org/Subclasses

                which has been deprecated and changed to...

              http://docs.scipy.org/doc/numpy/user/basics.subclassing.html
        '''
        if obj is None:
            return

        self.metadata = getattr(obj, 'metadata', {})

    def __array_wrap__(self, out_arr, context=None):
        return np.ndarray.__array_wrap__(self, out_arr, context)

    def __getitem__(self, index):
        """
        Override __getitem__ to return a simple (2, ) ndarray, rather than a
        Polygon object
        """
        return np.asarray(np.ndarray.__getitem__(self, index))

    def __eq__(self, other):
        if not isinstance(other, Polygon):
            # a Polygon is never equal to anything else
            return False
        else:
            return (np.array_equal(self, other) and
                    (self.metadata == other.metadata))

    def __ne__(self, other):
        return False if self == other else True

    def __str__(self):
        return ("Polygon with %i points.\nmetadata: %s" %
                (self.shape[0], self.metadata))

    def __repr__(self):
        msg = ["Polygon( [", ]
        pstr = []
        for point in self:
            try:
                pstr.append("[%s, %s]" % (point[0], point[1]))
            except IndexError:
                pass
        msg.append(",\n          ".join(pstr))
        msg.append("],\n         metadata=%s\n       )" % repr(self.metadata))
        return "".join(msg)

    @property
    def points(self):
        """
        the points as a regular np.ndarray
        """
        return np.asarray(self)

    @property
    def bounding_box(self):
        return BBox.fromPoints(self)

    @staticmethod
    def _scaling_fun(arr, scale):
        """
        scales and rounds -- does it all in place.
        """
        arr *= scale
        np.round(arr, out=arr)
        return arr

    def thin(self, scale):
        """
        Returns a new Polygon object, with the points thinned.

        :param scale: The scale to use: it is the ratio of world coords
                      (usually lat-lon degrees) to pixels.
        :type scale: (x_scale, y_scale): tuple of floats

        This is an algorithm designed for rendering. What it does
        is scale the points as you would to draw them (integer pixels).
        Then it removes any sequential duplicate points. Thus the rendered
        results should be exactly the same as if you rendered the pre-thinned
        polygons.

        Polygons that are reduced to 1 point are removed.

        If the polygon has teh first and last point the same, that property
        is preserved

        NOTE: in a sequence of close points, the first point is retained.
              Perhaps it would be better for the mean location of the
              sequence to be used instead? It should make no difference
              for rendering, but could make a difference for other purposes
        """
        scale = np.asarray(scale, dtype=np.float64)

        orig_poly = self
        sc_poly = self._scaling_fun(np.array(self), scale)
        prev_point = np.asarray(sc_poly[0])
        # special_case if last point matches first point
        last_same = 1 if np.array_equal(orig_poly[0], orig_poly[-1]) else 0
        thinned = [orig_poly[0]]
        for j in xrange(len(sc_poly)-last_same):
            point = sc_poly[j]
            if not np.array_equal(point, prev_point):
                thinned.append(orig_poly[j])
            prev_point = point
        if len(thinned) > 1:
            if last_same:
                thinned.append(orig_poly[0])
            return Polygon(thinned, metadata=orig_poly.metadata)
        else:
            return Polygon((), metadata=orig_poly.metadata)


class PolygonSet:
    # version that uses an Accumulator, rather than all that concatenating
    """
    A set of polygons (or polylines) stored as a single array of vertex data,
    and indexes into that array.
    """

    def __init__(self, data=None, dtype=np.float64):
        """
        create a new PolygonSet object

        if no data is passed in, and empty set is created.

        if data is passed in, it must a a tuple:
        (PointsArray, IndexArray, DataList)

        """
        self.dtype = dtype
        if data is None:
            self._PointsArray = np.zeros((0, 2), self.dtype)
            self._IndexArray = np.array((0,), dtype=np.int)
            self._MetaDataList = []
        else:
            self._PointsArray = np.array(data[0])
            self._IndexArray = np.array(data[1])
            self._MetaDataList = np.array(data[2])

    def append(self, polygon, metadata=None):

        """
        polygon should be a Polygon object or a  NX2 array (or something that
           can be turned into one)

        So that polygon[n,0] is the x coordinate of the nth point and
                polygon[n,1] is the y coordinate of the nth point

        """
        if metadata is None:
            metadata = getattr(polygon, 'metadata', {})
        polygon = np.asarray(polygon, dtype=self.dtype).reshape((-1, 2))
        # new method using resize() rather than concatanating
        # reduced test case run time from 10.3s to 1.85s !
        # self._PointsArray = np.r_[self._PointsArray, polygon]
        # self._IndexArray = np.r_[self._IndexArray,
        #                         (self._PointsArray.shape[0],)]

        old_length = self._PointsArray.shape[0]

        added_length = polygon.shape[0]
        self._PointsArray.resize((old_length+added_length, 2))
        self._PointsArray[-added_length:, :] = polygon

        self._IndexArray.resize((self._IndexArray.shape[0]+1))
        self._IndexArray[-1] = self._PointsArray.shape[0]
        self._MetaDataList.append(metadata)

    def _get_bounding_box(self):
        if len(self._PointsArray) > 0:
            return BBox.fromPoints(self._PointsArray)
        else:
            return None

    bounding_box = property(_get_bounding_box)

    def _get_total_num_points(self):
        return len(self._PointsArray)

    total_num_points = property(_get_total_num_points)

    def GetPointsData(self):
        """
        returns a copy of the points and indexes arrays
        """
        return (self._PointsArray.copy(), self._IndexArray.copy())

    def GetMetaData(self):
        """
        returns a (shallow) copy of the metadata list
        """
        return copy.copy(self._MetaDataList)

    def SetPointsData(self, PointData, MetaData=None):
        """

        SetPointsData(PointData)

        where PointData is a tuple of two NX2 arrays, or objects that can be
        converted to arrays:
        PointData = (PointsArray, IndexArray)

        Sets the data for a polygon set. Be careful with this one, it
        destroys all the current data, and doesn't check for a match
        between your PointsArray and IndexArray.

        The data type is preserved for the points, but it should probably be a
        float type.

        It can be useful for setting the data in one PolygonSet to the same as
        another set:
            set1.SetPointsData(set2.GetPointsData)

        A copy is made, so the two sets will be distinct

        """
        self._PointsArray = np.array(PointData[0], self.dtype)
        self._IndexArray = np.array(PointData[1], dtype=np.int)
        if MetaData is not None:
            self._DataArray = MetaData
        else:
            self._DataArray = [None] * len(self.PointsArray)

    def Copy(self):
        """
        returns a "deep copy" of the PolygonSet Object --
          i.e. it does not share any data with the original
        """
        cp = PolygonSet()
        cp._PointsArray = self._PointsArray.copy()
        cp._IndexArray = self._IndexArray.copy()
        cp._MetaDataList = copy.deepcopy(self._MetaDataList)

        return cp

    def TransformData(self, TransformFunction, args=(), kwargs={}):
        # fixme: if this was a ndarray subclass, it would "just work"
        """

        TransformData(Transform Function, args=(), kwargs={})

        Transforms the data for a polygon set. It applies the passed in
        Transform Function to all the points in the polygon set. The
        function needs to accept a NX2 NumPy array of Floats, and return
        a NX2 NumPy array of floats

        the optional arguments, args or kwargs are passed through to
        the TransformFuntion, so it is called as:

        NewPoints = TransformFunction(OldPoints, *args, **kwargs)

        """
        self._PointsArray = TransformFunction(self._PointsArray, *args,
                                              **kwargs)

    def __len__(self):
        # there is an extra index at the end, so that IndexArray[i+1] works
        return len(self._IndexArray) - 1

    def __getitem__(self, index):
        """
        returns a Polygon object
        """
        if index > (len(self._IndexArray) - 1):
            raise IndexError
        if index < 0:
            if index < -(len(self._IndexArray) - 1):
                raise IndexError
            index = len(self._IndexArray) - 1 + index
        poly = Polygon(self._PointsArray[self._IndexArray[index]:
                                         self._IndexArray[index + 1]],
                       metadata=self._MetaDataList[index],
                       dtype=self.dtype)
        return poly

    def __str__(self):
        return ("PolygonSet instance with %i polygons, %i total points"
                % (len(self), len(self._PointsArray)))

    def __repr__(self):
        """ same as __str__ -- not good but more informative than nothing"""
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, PolygonSet):
            # a PolygonSet is never equal to anything else
            return False
        else:
            return (np.array_equal(self._PointsArray, other._PointsArray) and
                    self._MetaDataList == other._MetaDataList)

    def __ne__(self, other):
        return False if self == other else True

    def thin(self, scale):
        """
        Returns a new PolygonSet object, with the points thinned.

        :param scale: The scale to use: it is the ratio of world coords
                      (usually lat-lon degrees) to pixels.
        :type scale: (x_scale, y_scale): tuple of floats

        This is an algorithm designed for rendering. What is does
        is scale the points as you would to draw them (integer pixels).
        Then it removes any sequential duplicate points. Thus the rendered
        results should be exactly the same as if you rendered the pre-thinned
        polygons.

        Polygons that are reduced to 1 point are removed.

        NOTE: in a sequence of close points, the first point is retained.
              Perhaps it would be better for the mean location of the
              sequence to be used instead? It should make no difference
              for rendering, but could make a difference for other purposes
        """
        new_polys = PolygonSet()
        for poly in self:
            poly = poly.thin(scale)
            if len(poly):
                new_polys.append(poly)
        return new_polys


def test():
    #  a test function

    p1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    p2 = p1 * 5

    set_ = PolygonSet()
    set_.append(p1)
    set_.append(p2)

    print set_[0]
    print set_[1]

    print "minimum is: ", set_.GetBoundingBox()[0]
    print "maximum is: ", set_.GetBoundingBox()[1]


if __name__ == "__main__":
    # run a test function
    test()
