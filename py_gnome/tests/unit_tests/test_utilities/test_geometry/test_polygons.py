## test in_place

"""
tests for the polygon module, part of the geometry package

designed to be run with nose
"""

from gnome.utilities.geometry.polygons import Polygon, PolygonSet

import numpy as np

p1 = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
p2 = p1 * 5


class Test_Polygon:

    def test_init(self):

        # all this needs to do is not raise an error

        Polygon(p1)

    def test_str(self):
        P = Polygon(p1, metadata={'name': 'a name'})
        print P
        assert str(P) \
            == "Polygon with 4 points.\nmetadata: {'name': 'a name'}"

    def test_repr(self):
        P = Polygon(p1, metadata={'name': 'a name'})
        print 'ppoints:', P.dtype, P.shape
        for p in P:
            print type(p), p
        print repr(P)
        assert repr(P) \
            == """Polygon( [[1.0, 2.0],
          [3.0, 4.0],
          [5.0, 6.0],
          [7.0, 8.0]],
         metadata={'name': 'a name'}
       )"""

    def test_list(self):

        # all this needs to do is not raise an error

        Polygon([(1, 2), (3, 4), (5, 6)])

    def test_index(self):
        P = Polygon([(1, 2), (3, 4), (5, 6)])
        assert P[1, 0] == 3.0 and P[1, 1] == 4.0

    def test_index2(self):
        P = Polygon([(1, 2), (3, 4), (5, 6)])
        assert np.array_equal(P[2], np.array((5, 6), dtype=np.float))

    def test_slice(self):
        P = Polygon([(1, 2), (3, 4), (5, 6)])
        print P[:2]
        assert P[:2] == Polygon([(1, 2), (3, 4)])     

    def test_metadata(self):
        m = {'name': 'a polygon', 'type': 'polyline'}
        P = Polygon([(1, 2), (3, 4), (5, 6)], metadata=m)
        assert P.metadata['name'] == 'a polygon'

    def test_copy(self):
        m = {'name': 'a polygon', 'type': 'polyline'}
        P1 = Polygon([(1, 2), (3, 4), (5, 6)], metadata=m)
        P2 = Polygon(P1)
        assert P2.metadata['name'] == 'a polygon'

    def test_data_copy(self):
        P1 = Polygon([(1, 2), (3, 4), (5, 6)])
        P2 = Polygon(P1, copy=True)
        P2[1] = (9, 10)
        assert tuple(P1[1]) == (3.0, 4.0)

    def test_data_view(self):
        P1 = Polygon([(1, 2), (3, 4), (5, 6)])
        P2 = Polygon(P1, copy=False)
        P2[1] = (9, 10)
        assert tuple(P1[1]) == (9.0, 10.0)

    def test_bounding_box(self):
        P = Polygon(p1)
        print P.bounding_box
        assert P.bounding_box == np.array([[1., 2.], [7., 8.]],
                dtype=np.float)

    def test_size_zero(self):
        P = Polygon( (), )
        assert len(P) == 0



class Test_PolygonSet:

    def test_append(self):

        # this passes as long as there is no error!

        poly_set = PolygonSet()
        poly_set.append(p1)
        poly_set.append(p2)

    def test_bbox(self):
        poly_set = PolygonSet()
        poly_set.append(p1)
        poly_set.append(p2)
        bb = np.array(((1., 2.), (35., 40.)), dtype=np.float)

        assert np.array_equal(poly_set.bounding_box, bb)

    def test_datatype(self):
        poly_set = PolygonSet(dtype=np.float32)
        poly_set.append(p1)
        poly_set.append(p2)
        assert poly_set.dtype == np.float32

    def test_datatype2(self):
        poly_set = PolygonSet(dtype=np.float32)
        poly_set.append(p1)
        poly_set.append(p2)
        print poly_set[0]
        print poly_set[0].dtype
        assert poly_set[0].dtype == np.float32

    def test_zero_length(self):
        poly_set = PolygonSet()

        assert len(poly_set) == 0

    # def test_pop(self):
    #    pass

#    def test_indexing1(self):
#        poly_set = PolygonSet(dtype=np.float32)
#        poly_set.append(p1)
#        poly_set.append(p2)
#        poly_set.append(p1)
#        print poly_set[-1]
#        assert array_equal(poly_set[-1], ff)
