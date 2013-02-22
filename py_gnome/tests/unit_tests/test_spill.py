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
    each spill class maintains a static dict of data types referenced by a 'property name'
    which correspond to the properties that LEs will have when released by the spill
    """
    sp = Spill()
    for k in ('positions', 'next_positions', 'last_water_positions', 'status_codes', 'spill_num',):
        assert k in sp.array_types

    sp = FloatingSpill()
    for k in ('positions', 'next_positions', 'last_water_positions', 'status_codes', 'spill_num', 'windages'):
        assert k in sp.array_types

    # check that the FloatingSpill did not affect the Spill Base class array types
    sp = Spill()
    for k in ('positions', 'next_positions', 'last_water_positions', 'status_codes', 'spill_num',):
        assert k in sp.array_types
    assert 'windages' not in sp.array_types


def test_deepcopy():
    """
    only tests that the spill_nums work -- not sure about anything else...

    test_spill_container does test some other issues.
    """
    sp1 = Spill()
    sp2 = copy.deepcopy(sp1)
    assert sp1 is not sp2

    #try deleting the copy, and see if any errors result
    del sp2
    del sp1

def test_copy():
    """
    only tests that the spill_nums work -- not sure about anything else...
    """
    sp1 = Spill()
    sp2 = copy.copy(sp1)
    assert sp1 is not sp2

    #try deleting the copy, and see if any errors result
    del sp1
    del sp2


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


class Test_SurfaceReleaseSpill(object):
    num_elements = 10
    start_position = (-128.3, 28.5, 0)
    release_time=datetime.datetime(2012, 8, 20, 13)
    timestep = 3600 # one hour in seconds

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
        timestep = 3600 # seconds
        arrays = sp.release_elements(self.release_time, timestep)
        assert arrays['positions'].shape == (10,3)
        assert np.alltrue( arrays['positions'] == self.start_position )
        assert np.alltrue( arrays['status_codes'] == basic_types.oil_status.in_water)

        assert sp.num_released == self.num_elements

        arrays = sp.release_elements(self.release_time + datetime.timedelta(10), timestep)
        assert arrays is None

        # reset and try again
        sp.rewind()
        assert sp.num_released == 0
        arrays = sp.release_elements(self.release_time - datetime.timedelta(10), timestep)
        assert arrays is None
        assert sp.num_released == 0

        arrays = sp.release_elements(self.release_time + datetime.timedelta(10), timestep)
        assert arrays['positions'].shape == (10,3)
        assert np.alltrue( arrays['positions'] == self.start_position )
        assert np.alltrue( arrays['status_codes'] == basic_types.oil_status.in_water)

    def test_cont_release(self):
        sp = SurfaceReleaseSpill(num_elements = 100,
                                 start_position = self.start_position,
                                 release_time = self.release_time,
                                 end_release_time = self.release_time + datetime.timedelta(hours=10),
                                 )
        timestep = 3600 # one hour in seconds

        # at exactly the release time -- ten get released
        arrays = sp.release_elements(self.release_time, timestep)
        assert arrays['positions'].shape == (10,3)

        # one hour into release -- ten more released
        arrays = sp.release_elements(self.release_time + datetime.timedelta(hours=1), timestep)
        assert arrays['positions'].shape == (10,3)
        assert sp.num_released == 20

        # 1-1/2 hours into release - 5 more
        arrays = sp.release_elements(self.release_time + datetime.timedelta(hours=2), timestep/2)
        assert arrays['positions'].shape == (5,3)
        assert sp.num_released == 25

        # at end -- rest should be released:
        arrays = sp.release_elements(self.release_time + datetime.timedelta(hours=10), timestep)
        assert arrays['positions'].shape == (75,3)
        assert sp.num_released == 100

        sp.rewind()

        ## 360 second time step: first LE
        arrays = sp.release_elements(self.release_time, 360)
        assert arrays['positions'].shape == (1,3)
        assert np.alltrue( arrays['positions'] == self.start_position )

        ## 720 seconds: one more
        arrays = sp.release_elements(self.release_time + datetime.timedelta(seconds=360), 360)
        assert arrays['positions'].shape == (1,3)
        assert np.alltrue( arrays['positions'] == self.start_position )
        assert sp.num_released == 2

    def test_inst_line_release(self):
        sp = SurfaceReleaseSpill(num_elements = 11, # so it's easy to compute where they should be!
                                  start_position = (-128.0, 28.0, 0),
                                  release_time = self.release_time,
                                  end_position = (-129.0, 29.0, 0)
                                  )
        timestep = 600 # ten minutes in seconds         
        arrays = sp.release_elements(self.release_time + datetime.timedelta(hours=1), timestep)

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
        timestep = 100 * 60       
        # first the full release over one time step
        arrays = sp.release_elements(self.release_time, timestep )

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

        # release at release time with time step of 1/10 of release_time
        arrays = sp.release_elements(self.release_time, 10*60)

        assert arrays['positions'].shape == (10,3)
        assert np.alltrue( arrays['status_codes'] == basic_types.oil_status.in_water)
        assert np.array_equal( arrays['positions'][:,0], np.linspace(-128, -128.1, 10))
        assert np.array_equal( arrays['positions'][:,1], np.linspace(28, 28.1, 10))
        assert sp.num_released == 10

        # second time step release:
        arrays = sp.release_elements(self.release_time + datetime.timedelta(minutes=10), 10*60 )

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
        timestep = delta_t.total_seconds()
        positions = np.zeros((0,3), dtype=np.float64)
        # end after release
        while time < self.release_time + datetime.timedelta(minutes=100):
            arrays = sp.release_elements(time, timestep)
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
        timestep = delta_t.total_seconds()
        positions = np.zeros((0,3), dtype=np.float64)
        # end after release
        while time < self.release_time + datetime.timedelta(minutes=100):
            arrays = sp.release_elements(time, timestep)
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
        timestep = delta_t.total_seconds()
        positions = np.zeros((0,3), dtype=np.float64)
        # end after release
        while time < self.release_time + datetime.timedelta(minutes=100):
            arrays = sp.release_elements(time, timestep)
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

    def test_cont_not_valid_times(self):        
        with pytest.raises(ValueError):
            sp = SurfaceReleaseSpill(num_elements = 100,
                                     start_position = self.start_position,
                                     release_time = self.release_time,
                                     end_release_time = self.release_time - datetime.timedelta(seconds=1),
                                     )
            print sp

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
    data = sp.release_elements(release_time, time_step=600)

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
    data = sp.release_elements(release_time, time_step=600)

    assert data['positions'].shape == (4,3)
    data = sp.release_elements(release_time+datetime.timedelta(hours=1), time_step=600)



if __name__ == "__main__":
    test_uncertain_copy()
    #test_reset_array_types()


