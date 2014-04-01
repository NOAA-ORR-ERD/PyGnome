#!/usr/bin/env python

import collections

import numpy
np = numpy


class IArray(object):
    '''
       An interpolated array based on numpy
       This contains some added functionality from the standard np.interp()
       method
    '''
    def __init__(self, points, left=None, right=None, method='linear'):
        '''
           :param points: point data that we would like to interpolate
           :type points: Sequence of 2-tuple items ((x, y),
                                                    ...
                                                    )
           :param left: Value to return for i < x[0], default is y[0].
           :param right: Value to return for i > x[-1], default is y[-1].
           :param method: interpolation method to use.
           :type method: a string in {'linear', 'nearest',
                                      'leftmost', 'rightmost'}
        '''
        self.points = sorted(points)
        self.x = np.array([p[0] for p in self.points])
        self.y = np.array([p[1] for p in self.points])
        self.left = left
        self.right = right

        methods = {'linear': self.interp,
                   'nearest': self.nearest,
                   'leftmost': self.leftmost,
                   'rightmost': self.rightmost
                   }
        self.method = methods[method]

    def _get_bounding(self, coords):
        if self.method == self.leftmost:
            rightside = np.searchsorted(self.x, coords, side='right')
        else:
            rightside = np.searchsorted(self.x, coords)

        leftside = rightside - 1
        if isinstance(rightside, np.ndarray):
            leftside[leftside < 0] = 0
            rightside[rightside > (self.x.shape[0] - 1)] = (self.x.shape[0] - 1)
        else:
            # scalar handling
            if leftside < 0:
                leftside = 0
            if rightside > (self.x.shape[0] - 1):
                rightside = (self.x.shape[0] - 1)

        return leftside, rightside

    def nearest(self, coords):
        """
           Interpolate the curve using nearest neighbor method.
        """
        leftside, rightside = self._get_bounding(coords)
        return np.where(np.abs(coords - self.x[leftside]) < np.abs(self.x[rightside] - coords),
                       self.y[leftside], self.y[rightside])

    def leftmost(self, coords):
        """
           Interpolate the curve using leftmost neighbor method.
        """
        leftside, rightside = self._get_bounding(coords)
        return self.y[leftside]

    def rightmost(self, coords):
        """
           Interpolate the curve using rightmost neighbor method.
        """
        leftside, rightside = self._get_bounding(coords)
        return self.y[rightside]

    def interp(self, coords):
        return np.interp(coords, self.x, self.y, self.left, self.right)

    def __getitem__(self, coords):
        return self.method(coords)


if __name__ == '__main__':
    points = ((1, 0), (5, 10), (10, 0))

    print 'Testing IArray()...'
    print 'Loading Points:', points
    table = IArray(points)

    for i in range(1, 11, 1):
        print '%d: %f' % (i, table[i])

    i = range(1, 11, 1)
    print '%s: %s' % (i, table[i])

    print '\n', 3.2, table[3.2]

    print 'Testing Leftmost Neighbor Interpolated Array()...'
    print 'Loading Points:', points
    table = IArray(points, method='leftmost')

    for i in range(12):
        print '%d: %f' % (i, table[i])

    print '\nTesting our table limits...'
    assert table[0] == 0
    assert table[1] == 0
    assert table[5] == 10
    assert table[10] == 0
    assert table[30] == 0
    assert tuple(table[(0, 1, 5, 10, 30)]) == (0, 0, 10, 0, 0)

    # test our interpolations
    print '\nTesting our interpolation behavior...'
    assert table[2.5] == 0
    assert table[7.5] == 10

    print 'testing our nearest neighbor stuff'
    table = IArray(points, method='nearest')
    coords = np.linspace(-2., 12., 15)
    assert tuple(table[coords]) == (0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0)

    print 'testing our leftmost interpolation'
    table = IArray(points, method='leftmost')
    coords = np.linspace(-2., 12., 15)
    print coords
    print table[coords]
    assert tuple(table[coords]) == (0, 0, 0, 0, 0, 0, 0, 10, 10, 10, 10, 10, 0, 0, 0)

    print 'testing our rightmost interpolation'
    table = IArray(points, method='rightmost')
    coords = np.linspace(-2., 12., 15)
    print coords
    print table[coords]
    assert tuple(table[coords]) == (0, 0, 0, 0, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0, 0)
    assert table[1.01] == 10  # immediately after the previous point
    assert table[4.99] == 10
    assert table[5.0] == 10  # inclusive
    assert table[5.01] == 0  # immediately after the current point
