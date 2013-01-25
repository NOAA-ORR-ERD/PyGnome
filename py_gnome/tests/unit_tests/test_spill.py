#!/usr/bin/env python

"""
Tests the spill code.
"""

import datetime
import copy

import pytest

import numpy as np

from gnome import basic_types
from gnome.spill import Spill, FloatingSpill, SurfaceReleaseSpill, SpatialReleaseSpill


def test_init_Spill():
    """
    the base class does not do much
    """
    sp = Spill()

    assert  sp.spill_num > 0


def test_set_spill_num():
    """
    spill_nums should get set, and stay unique as you delete and create spills
    """

    spills = [Spill() for i in range(10)]

    # the spill_nums are unique
    assert len( set([spill.spill_num for spill in spills]) ) == len(spills)

    #delete and create a few:
    del spills[3]
    del spills[5]

    spills.extend(  [FloatingSpill() for i in range(5)] ) 

    # spill_nums still unique
    assert len( set([spill.spill_num for spill in spills]) ) == len(spills)

    del spills[10]
    del spills[4]

    spills.extend(  [SurfaceReleaseSpill(5, (0,0,0), None ) for i in range(5)] ) 

    # spill_nums still unique
    assert len( set([spill.spill_num for spill in spills]) ) == len(spills)

    #print [spill.spill_num for spill in spills]
    #assert False

def test_deepcopy():
    """
    only tests that the spill_nums work -- not sure about anything else...

    test_spill_container does test some other issues.
    """
    spill1 = Spill() 
    spill2 = copy.deepcopy(spill1)
    assert spill1 is not spill2
    assert spill1.spill_num != spill2.spill_num

    #try deleting the copy, and see if any errors result
    del spill2
    del spill1

def test_copy():
    """
    only tests that the spill_nums work -- not sure about anything else...
    """
    spill1 = Spill() 
    spill2 = copy.copy(spill1)
    assert spill1 is not spill2
    assert spill1.spill_num != spill2.spill_num
    #try deleting the copy, and see if any errors result
    del spill1
    del spill2


def test_uncertain_copy():
    """
    only tests a few things...
    """
    spill = SurfaceReleaseSpill(num_elements=100,
                                start_position = (28, -78, 0.0),
                                release_time = datetime.datetime.now(),
                                end_position = (29, -79, 0.0),
                                end_release_time = datetime.datetime.now() + datetime.timedelta(hours=24),
                                windage_range = (0.02, 0.03),
                                windage_persist = 0,)

    u_spill = spill.uncertain_copy() 

    assert u_spill is not spill
    assert u_spill.spill_num == spill.spill_num
    assert np.array_equal(u_spill.start_position, spill.start_position)
    del spill
    del u_spill
    #assert False



def test_new_elements():
    """
    see if creating new elements works
    """
    sp = Spill()

    arrays = sp.create_new_elements(3)

    print arrays
    for array in arrays.values():
        assert len(array) == 3
    # what else to test???

def test_FloatingSpill():
    """
    see if the right arrays get created
    """
    sp = FloatingSpill()
    data = sp.create_new_elements(10)
    assert 'windages' in data

    assert data['status_codes'].shape == (10,)
    assert data['positions'].shape == (10,3)
    assert np.alltrue( data['status_codes'] == basic_types.oil_status.in_water )


def test_reset_array_types():
    """
    tests to make sure that after resetting, only arrays that are
    used by existing spills are created

    NOTE: This test is sensitive to other tests 
          I suspect that when other tests have been run, the test harness
          may keep references around that defeats this. So test this by itself
    """
    sp1 = Spill()
    sp1.reset() # make sure that we're reset from previous tests

    sp2 = FloatingSpill()
    sp3 = Spill()
    sp4 = FloatingSpill()
    sp5 = Spill()

    arrays = sp1.create_new_elements(1)
    assert 'windages' in arrays

    # delete the FloatingSpill
    del sp2
    # windages still there
    arrays = sp1.create_new_elements(1)
    assert 'windages' in arrays

    sp1.reset()
    # windages still there
    arrays = sp1.create_new_elements(1)
    assert 'windages' in arrays

    del sp4
    sp5.reset()
    #windages should no longer be there
    #print Spill._Spill__all_subclasses
    arrays = sp1.create_new_elements(1)
    print "This test can fail if others have been run before it -- leaving dangling references"
    print "Fix other tests first"
    assert 'windages' not in arrays


class Test_SurfaceReleaseSpill():
    num_elements = 10
    start_position = (-128.3, 28.5, 0)
    release_time=datetime.datetime(2012, 8, 20, 13)

    def test_init(self):
        sp = SurfaceReleaseSpill(num_elements = self.num_elements,
                                 start_position = self.start_position,
                                 release_time = self.release_time,
                                 )
        arrays = sp.create_new_elements(10)
        assert arrays['status_codes'].shape == (10,)
        assert arrays['positions'].shape == (10,3)
        assert np.alltrue( arrays['status_codes'] == basic_types.oil_status.in_water)

    def test_inst_release(self):
        sp = SurfaceReleaseSpill(num_elements = self.num_elements,
                                 start_position = self.start_position,
                                 release_time = self.release_time,
                                 )
        arrays = sp.release_elements(self.release_time)
        assert arrays['positions'].shape == (10,3)
        assert np.alltrue( arrays['positions'] == self.start_position )
        assert np.alltrue( arrays['status_codes'] == basic_types.oil_status.in_water)

        assert sp.num_released == self.num_elements

        arrays = sp.release_elements(self.release_time + datetime.timedelta(10))
        assert arrays is None

        # reset and try again
        sp.reset()
        assert sp.num_released == 0
        arrays = sp.release_elements(self.release_time - datetime.timedelta(10))
        assert arrays is None
        assert sp.num_released == 0

        arrays = sp.release_elements(self.release_time + datetime.timedelta(10))
        assert arrays['positions'].shape == (10,3)
        assert np.alltrue( arrays['positions'] == self.start_position )
        assert np.alltrue( arrays['status_codes'] == basic_types.oil_status.in_water)

        # assert arrays['status_codes'].shape == (10,)
        # assert arrays['positions'].shape == (10,3)
        # assert np.alltrue( arrays['status_codes'] == basic_types.oil_status.in_water)

    def test_cont_release(self):
        sp = SurfaceReleaseSpill(num_elements = 100,
                                 start_position = self.start_position,
                                 release_time = self.release_time,
                                 end_release_time = self.release_time + datetime.timedelta(hours=10),
                                 )
        # at exactly the release time -- none get released
        arrays = sp.release_elements(self.release_time)
        assert arrays is None

        # one hour into release
        arrays = sp.release_elements(self.release_time + datetime.timedelta(hours=1))
        assert arrays['positions'].shape == (10,3)

        # 1-1/2 hours into release - 5 more
        arrays = sp.release_elements(self.release_time + datetime.timedelta(hours=1.5))
        assert arrays['positions'].shape == (5,3)
        assert sp.num_released == 15

        # at end -- rest should be released:
        arrays = sp.release_elements(self.release_time + datetime.timedelta(hours=10))
        assert arrays['positions'].shape == (85,3)
        assert sp.num_released == 100

        sp.reset()

        ## 1 second after start: none yet
        arrays = sp.release_elements(self.release_time + datetime.timedelta(seconds=1))
        assert arrays is None

        ## 300 seconds: still none
        arrays = sp.release_elements(self.release_time + datetime.timedelta(seconds=300))
        assert arrays is None

        ## 360 seconds: first LE
        arrays = sp.release_elements(self.release_time + datetime.timedelta(seconds=360))
        assert arrays['positions'].shape == (1,3)
        assert np.alltrue( arrays['positions'] == self.start_position )

        ## 300 seconds again: shouldn't crash
        arrays = sp.release_elements(self.release_time + datetime.timedelta(seconds=00))
        assert arrays is None

        ## 600 seconds: no more yet
        arrays = sp.release_elements(self.release_time + datetime.timedelta(seconds=600))
        assert arrays is None

        ## 720 seconds: one more
        arrays = sp.release_elements(self.release_time + datetime.timedelta(seconds=720))
        assert arrays['positions'].shape == (1,3)
        assert np.alltrue( arrays['positions'] == self.start_position )

    def test_inst_line_release(self):
        sp = SurfaceReleaseSpill(num_elements = 11, # so it's easy to compute where they should be!
                                 start_position = (-128.0, 28.0, 0),
                                 release_time = self.release_time,
                                 end_position = (-129.0, 29.0, 0)
                                 )
        
        arrays = sp.release_elements(self.release_time + datetime.timedelta(hours=1))
        
        assert arrays['positions'].shape == (11,3)
        assert np.alltrue( arrays['status_codes'] == basic_types.oil_status.in_water)
        assert np.array_equal( arrays['positions'][:,0], np.linspace(-128, -129, 11) )
        assert np.array_equal( arrays['positions'][:,1], np.linspace(28, 29, 11) )
        
        assert sp.num_released == 11

    def test_cont_line_release1(self):
        """
        testing a release that is releasing while moving over time

        In this one it all gets released in the first time step.
        """
        sp = SurfaceReleaseSpill(num_elements = 11, # so it's easy to compute where they should be!
                                 start_position = (-128.0, 28.0, 0),
                                 release_time = self.release_time,
                                 end_position = (-129.0, 29.0, 0),
                                 end_release_time = self.release_time + datetime.timedelta(minutes=100)
                                 )
        
        # first the full release over one time step
        arrays = sp.release_elements(self.release_time + datetime.timedelta(minutes=200) )
        
        assert arrays['positions'].shape == (11,3)
        assert np.alltrue( arrays['status_codes'] == basic_types.oil_status.in_water)
        assert np.array_equal( arrays['positions'][:,0], np.linspace(-128, -129, 11) )
        assert np.array_equal( arrays['positions'][:,1], np.linspace(28, 29, 11) )
        
        assert sp.num_released == 11

    def test_cont_line_release2(self):
        """
        testing a release that is releasing while moving over time

        first timestep, timestep less than release-time
        """
        sp = SurfaceReleaseSpill(num_elements = 100, 
                                 start_position = (-128.0, 28.0, 0),
                                 release_time = self.release_time,
                                 end_position = (-129.0, 29.0, 0),
                                 end_release_time = self.release_time + datetime.timedelta(minutes=100)
                                 )
        
        # release after 1/10 of release_time
        arrays = sp.release_elements(self.release_time + datetime.timedelta(minutes=10) )
        
        assert arrays['positions'].shape == (10,3)
        assert np.alltrue( arrays['status_codes'] == basic_types.oil_status.in_water)
        assert np.array_equal( arrays['positions'][:,0], np.linspace(-128, -128.1, 10))
        assert np.array_equal( arrays['positions'][:,1], np.linspace(28, 28.1, 10))
        assert sp.num_released == 10

        # second time step release:
        arrays = sp.release_elements(self.release_time + datetime.timedelta(minutes=20) )
        
        assert arrays['positions'].shape == (10,3)
        assert np.alltrue( arrays['status_codes'] == basic_types.oil_status.in_water)
        assert np.array_equal( arrays['positions'][:,0], np.linspace(-128.1, -128.2, 11)[1:])
        assert np.array_equal( arrays['positions'][:,1], np.linspace(28.1, 28.2, 11)[1:])
        assert sp.num_released == 20

    def test_cont_line_release3(self):
        """
        testing a release that is releasing while moving over time

        making sure it's right for the full release -- mutiple elements per step
        """
        sp = SurfaceReleaseSpill(num_elements = 50, 
                                 start_position = (-128.0, 28.0, 0),
                                 release_time = self.release_time,
                                 end_position = (-129.0, 30.0, 0),
                                 end_release_time = self.release_time + datetime.timedelta(minutes=50)
                                 )
        
        #start before release
        time = self.release_time - datetime.timedelta(minutes=10)
        delta_t = datetime.timedelta(minutes=10)
        positions = np.zeros((0,3), dtype=np.float64)
        # end after release
        while time < self.release_time + datetime.timedelta(minutes=100):
            arrays = sp.release_elements(time)
            if arrays is not None:
                positions = np.r_[positions, arrays['positions'] ]
            time += delta_t
        assert positions.shape == (50,3)
        assert np.array_equal(positions[0], (-128.0, 28.0, 0) )
        assert np.array_equal(positions[-1], (-129.0, 30.0, 0) )
        # check if they are all close to the same line (constant slope)
        diff = np.diff(positions[:,:2], axis=0)
        assert np.alltrue( np.abs(np.diff ( diff[:,0] / diff[:,1] ) ) < 1e-10 )

    def test_cont_line_release4(self):
        """
        testing a release that is releasing while moving over time

        making sure it's right for the full release -- less than one elements per step
        """
        sp = SurfaceReleaseSpill(num_elements = 10, 
                                 start_position = (-128.0, 28.0, 0),
                                 release_time = self.release_time,
                                 end_position = (-129.0, 31.0, 0),
                                 end_release_time = self.release_time + datetime.timedelta(minutes=50)
                                 )
        
        #start before release
        time = self.release_time - datetime.timedelta(minutes=10)
        delta_t = datetime.timedelta(minutes=2)
        positions = np.zeros((0,3), dtype=np.float64)
        # end after release
        while time < self.release_time + datetime.timedelta(minutes=100):
            arrays = sp.release_elements(time)
            print arrays
            if arrays is not None:
                positions = np.r_[positions, arrays['positions'] ]
            time += delta_t
        assert positions.shape == (10,3)
        assert np.array_equal(positions[0], (-128.0, 28.0, 0) )
        assert np.array_equal(positions[-1], (-129.0, 31.0, 0) )
        # check for monotonic 
        assert np.alltrue(np.sign(np.diff(positions[:,:2], axis=0)) == (-1, 1) )# check monotonic 
        # check if they are all close to the same line (constant slope)
        diff = np.diff(positions[:,:2], axis=0)
        assert np.alltrue( np.abs(np.diff ( diff[:,0] / diff[:,1] ) ) < 1e-10 )

    positions = [( (128.0, 2.0, 0), (128.0, -2.0, 0)), # south
                 ( (128.0, 2.0, 0), (128.0, 4.0, 0)), # north
                 ( (128.0, 2.0, 0), (125.0, 2.0, 0)), # west
                 ( (-128.0, 2.0, 0), (-120.0, 2.0, 0) ), # east
                 ( (-128.0, 2.0, 0), (-120.0, 2.01, 0) ), # almost east
                 ]
    @pytest.mark.parametrize( ('start_position', 'end_position'), positions )
    def test_south_line(self, start_position, end_position):
        """
        testing a line release to the north
        making sure it's right for the full release -- mutiple elements per step
        """
        sp = SurfaceReleaseSpill(num_elements = 50, 
                                 start_position = start_position,
                                 release_time = self.release_time,
                                 end_position = end_position,
                                 end_release_time = self.release_time + datetime.timedelta(minutes=50)
                                 )
        
        #start before release
        time = self.release_time - datetime.timedelta(minutes=10)
        delta_t = datetime.timedelta(minutes=10)
        positions = np.zeros((0,3), dtype=np.float64)
        # end after release
        while time < self.release_time + datetime.timedelta(minutes=100):
            arrays = sp.release_elements(time)
            if arrays is not None:
                positions = np.r_[positions, arrays['positions'] ]
            time += delta_t
        assert positions.shape == (50,3)
        assert np.array_equal(positions[0], start_position )
        assert np.array_equal(positions[-1], end_position )
        # check if they are all close to the same line (constant slope)
        diff = np.diff(positions[:,:2], axis=0)
        if start_position[1] == end_position[1]:
            #horizontal line -- infinite slope
            assert np.alltrue( diff[:,1]  == 0 )
        else:
            assert np.alltrue( np.abs(np.diff ( diff[:,0] / diff[:,1] ) ) < 1e-8 )

    def cont_not_valid_times(self):        
        with pytest.raises(ValueError):
            sp = SurfaceReleaseSpill(num_elements = 100,
                                     start_position = self.start_position,
                                     release_time = self.release_time,
                                     end_release_time = self.release_time - datetime.timedelta(seconds=1),
                                     )

def test_SpatialReleaseSpill():
    """
    see if the right arrays get created
    """
    start_positions = ( (0.0,   0.0,   0.0 ),
                        (28.0, -75.0,  0.0 ),
                        (-15,    12,    4.0),
                        ( 80,    -80,  100.0),
                        )
    release_time = datetime.datetime(2012,1,1,1)
    sp = SpatialReleaseSpill(start_positions,
                             release_time,
                             windage_range=(0.01, 0.04),
                             windage_persist=900,
                             )
    data = sp.release_elements(release_time)

    assert 'windages' in data

    assert data['status_codes'].shape == (4,)
    assert data['positions'].shape == (4,3)
    assert np.alltrue( data['status_codes'] == basic_types.oil_status.in_water )

def test_SpatialReleaseSpill2():
    """
    make sure they don't release elements twice
    """
    start_positions = ( (0.0,   0.0,   0.0 ),
                        (28.0, -75.0,  0.0 ),
                        (-15,    12,    4.0),
                        ( 80,    -80,  100.0),
                        )
    release_time = datetime.datetime(2012,1,1,1)
    sp = SpatialReleaseSpill(start_positions,
                             release_time,
                             windage_range=(0.01, 0.04),
                             windage_persist=900,
                             )
    data = sp.release_elements(release_time)

    assert data['positions'].shape == (4,3)
    data = sp.release_elements(release_time+datetime.timedelta(hours=1))



if __name__ == "__main__":
    test_uncertain_copy()
    #test_reset_array_types()


