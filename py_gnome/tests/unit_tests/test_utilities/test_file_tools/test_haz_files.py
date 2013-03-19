#!/usr/bin/env python
"""
tests for haz_files module

(only bna reader tests for now)

designed for py.test

"""

import os
import numpy as np
from gnome.utilities.file_tools import haz_files
import pytest

## NOTE: according to:
## http://www.softwright.com/faq/support/boundary_file_bna_format.html
## polygons in BNA shouls have the first and last popint the same. 
## these tests (Nor the code) do not enforce that, but rather add the extra
## point if it's not there


#  write a simple test bna file
file('test.bna', 'w').write( \
'''"Another Name","Another Type", 7
-81.531753540039,31.134635925293
-81.531150817871,31.134529113769
-81.530662536621,31.134353637695
-81.530502319336,31.134126663208
-81.530685424805,31.133970260620
-81.531112670898,31.134040832519
-81.531753540039,31.134635925293
"A third 'name'","6", 5
-81.522369384766,31.122062683106
-81.522109985352,31.121908187866
-81.522010803223,31.121685028076
-81.522254943848,31.121658325195
-81.522483825684,31.121797561646
"A name with, a comma","1", 9
-81.523277282715,31.122261047363
-81.522987365723,31.121982574463
-81.523200988770,31.121547698975
-81.523361206055,31.121408462524
-81.523818969727,31.121549606323
-81.524078369141,31.121662139893
-81.524009704590,31.121944427490
-81.523925781250,31.122068405151
-81.523277282715,31.122261047363
"A polyline","something", -4
-81.523277282715,31.122261047363
-81.522987365723,31.121982574463
-81.523200988770,31.121547698975
-81.523361206055,31.121408462524
"A point","3", 1
-81.523277282715,31.122261047363
"extra space around commas" , "1" , 9
-81.523277282715,31.122261047363
-81.522987365723,31.121982574463
-81.523200988770,31.121547698975
-81.523361206055,31.121408462524
-81.523818969727,31.121549606323
-81.524078369141,31.121662139893
-81.524009704590,31.121944427490
-81.523925781250,31.122068405151
-81.523277282715,31.122261047363
''')    

#  write a simple test bna file with invalid data
file('test_bad.bna', 'w').write( \
'''"An too-small polygon","Another Type", 2
-81.531753540039,31.134635925293
-81.531150817871,31.134529113769
"A third 'name'","6", 5
-81.522369384766,31.122062683106
-81.522109985352,31.121908187866
-81.522010803223,31.121685028076
-81.522254943848,31.121658325195
-81.522483825684,31.121797561646
''')

class Test_bna_list:
    polys = haz_files.ReadBNA("test.bna")

    def test_length(self):
        assert len(self.polys) == 6
    
    def test_type(self):
        assert self.polys[0][1] == 'polygon'
        assert self.polys[1][1] == 'polygon'
        assert self.polys[2][1] == 'polygon'
        assert self.polys[3][1] == 'polyline'
        assert self.polys[4][1] == 'point'

    def test_name(self):
        assert self.polys[0][2] == 'Another Name'
        assert self.polys[1][2] == "A third 'name'"
        assert self.polys[2][2] == 'A name with, a comma'

    def test_sname(self):
        print self.polys[0]
        assert self.polys[0][3] == 'Another Type'
        assert self.polys[1][3] == '6'
        assert self.polys[2][3] == '1'
        
    def test_points(self):
        assert np.array_equal( self.polys[0][0],
                               np.array(((-81.531753540039,31.134635925293),
                                         (-81.531150817871,31.134529113769),
                                         (-81.530662536621,31.134353637695),
                                         (-81.530502319336,31.134126663208),
                                         (-81.530685424805,31.133970260620),
                                         (-81.531112670898,31.134040832519),
                                         )))
    def test_end_points(self):
        for p in self.polys:
            if p[1] == 'polygon' or p[1] == 'polyline':
                assert not np.array_equal( p[0][0], p[0][-1] )

    def test_end_points_point(self):
        for p in self.polys:
            if p[1] == 'point':
                assert p[0].shape == (1, 2)

    def test_dtype(self):
        polys = haz_files.ReadBNA("test.bna", polytype = "list", dtype=np.float32)
        for p in polys:
            assert p[0].dtype == np.float32


class Test_bna_polygonset:    
    polys = haz_files.ReadBNA("test.bna", "PolygonSet")

    def test_length(self):
        assert len(self.polys) == 6

    def test_type(self):
        print self.polys[0]
        assert self.polys[0].metadata[0] == 'polygon'
        assert self.polys[3].metadata[0] == 'polyline'
        assert self.polys[4].metadata[0] == 'point'

    def test_name(self):
        assert self.polys[0].metadata[1] == 'Another Name'
        assert self.polys[1].metadata[1] == "A third 'name'"
        assert self.polys[2].metadata[1] == 'A name with, a comma'

    def test_sname(self):
        print self.polys[0]
        assert self.polys[0].metadata[2] == 'Another Type'
        assert self.polys[1].metadata[2] == '6'
        assert self.polys[2].metadata[2] == '1'

    def test_dtype(self):
        polys = haz_files.ReadBNA("test.bna", polytype = "PolygonSet", dtype=np.float32)
        for p in polys:
            assert p.dtype == np.float32

        