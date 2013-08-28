#!/usr/bin/env python

"""
Tests the spill code.
"""

from datetime import datetime, timedelta
import copy

import pytest

import numpy as np

from gnome.spill import (Spill, FloatingSpill, PointSourceSurfaceRelease, SpatialRelease,
                         SubsurfaceSpill, SubsurfaceRelease, SpatialRelease, OilProps)


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
    spill = PointSourceSurfaceRelease(num_elements=100,
                               start_position=(28, -78, 0.0),
                               release_time=datetime.now(),
                               end_position=(29, -79, 0.0),
                               end_release_time=datetime.now() + timedelta(hours=24),
                               windage_range=(0.02, 0.03),
                               windage_persist=0,)

    u_spill = spill.uncertain_copy()

    assert u_spill is not spill
    assert np.array_equal(u_spill.start_position, spill.start_position)
    del spill
    del u_spill


# use units of m^3 for volume
@pytest.mark.parametrize(("sp_obj", "num_elems"),
                         [(Spill(), 0), # volume = 0 for this case 
                          (FloatingSpill( volume=1e-6, volume_units='m^3' ), 0),
                          (PointSourceSurfaceRelease( 5, (0.0, 0.0, 0.0), 
                                                      datetime.now(), 
                                                      volume=1e-5, 
                                                      volume_units='m^3' ), 5),
                          (SubsurfaceSpill( volume=1e-5, volume_units='m^3' ), 0),
                          (SubsurfaceRelease( 5, (0.0, 0.0, 0.0), 
                                              datetime.now(), 
                                              volume=1e-4, 
                                              volume_units='m^3' ), 5),
                          (SpatialRelease( (0.0, 0.0, 0.0), 
                                           datetime.now(), 
                                           volume=1e-3, 
                                           volume_units='m^3' ), 5)])
def test_create_new_elements(sp_obj, num_elems):
    """
    see if creating new elements works
    """
    arrays = sp_obj.create_new_elements(num_elems)

    for name, array in arrays.iteritems():
        assert name in sp_obj.array_types
        assert len(array) == num_elems
        
        if num_elems > 0:
            if name == 'mass':
                exp_mass = sp_obj.oil_props.get_density('g/cm^3')*sp_obj.get_volume('cm^3')
                assert abs( np.sum( np.sum(array) ) - exp_mass) < 1e-10
            if name == 'positions':
                assert np.all( array == (0., 0., 0.) )
    
            #===================================================================
            # print arrays['mass']
            # print
            #===================================================================
            
    assert sp_obj.oil_props.name == 'oil_conservative'
    assert sp_obj.oil_props.get_density('g/cm^3') == 1    # same as water


class Test_PointSourceSurfaceRelease(object):
    num_elements = 10
    start_position = (-128.3, 28.5, 0)
    release_time = datetime(2012, 8, 20, 13)
    timestep = 3600  # one hour in seconds

    def test_init(self):
        sp = PointSourceSurfaceRelease(num_elements=self.num_elements,
                                start_position=self.start_position,
                                release_time=self.release_time,
                                )

        arrays = sp.create_new_elements(10)
        assert arrays['positions'].shape == (10, 3)
        assert np.alltrue(arrays['positions'] == sp.array_types['positions'].initial_value)

    def test_model_run_after_release(self):
        """
        Tests that the spill doesn't release anything if the first call
        to release_elements is after the release time.
        This so that if the user sets the model start time after the spill,
        they don't get anything.
        """
        sp = PointSourceSurfaceRelease(num_elements = self.num_elements,
                                 start_position = self.start_position,
                                 release_time = self.release_time,
                                 )
        data = sp.release_elements(self.release_time+timedelta(hours=1), time_step = 30*60)
        assert data is None

        # try again later
        data = sp.release_elements(self.release_time + timedelta(hours=2),
                                   time_step=30 * 60)
        assert data is None

        # rewind and it should work
        sp.rewind()
        arrays = sp.release_elements(self.release_time, time_step=30 * 60)
        assert arrays['positions'].shape == (10, 3)
        assert np.alltrue(arrays['positions'] == self.start_position)

        assert sp.num_released == self.num_elements

    def test_model_skips_over_release_time(self):
        """
        Tests that the spill doesn't release anything if the first call
        to release_elements is after the release time.
        This so that if the user sets the model start time after the spill,
        they don't get anything.
        """
        sp = PointSourceSurfaceRelease(num_elements = self.num_elements,
                                 start_position = self.start_position,
                                 release_time = self.release_time,
                                 )
        print "release_time:", self.release_time
        timestep = 360  # seconds

        #right before the release
        arrays = sp.release_elements(self.release_time - timedelta(seconds=360),
                                     timestep)
        assert arrays is None

        #right after the release
        arrays = sp.release_elements(self.release_time + timedelta(seconds=1),
                                     timestep)
        assert arrays['positions'].shape == (10, 3)
        assert np.alltrue(arrays['positions'] == self.start_position)

        assert sp.num_released == self.num_elements

    def test_inst_release(self):
        sp = PointSourceSurfaceRelease(num_elements=self.num_elements,
                                 start_position=self.start_position,
                                 release_time=self.release_time,
                                 )
        timestep = 3600 # seconds
        arrays = sp.release_elements(self.release_time, timestep)
        assert arrays['positions'].shape == (10, 3)
        assert np.alltrue(arrays['positions'] == self.start_position)

        assert sp.num_released == self.num_elements

        arrays = sp.release_elements(self.release_time + timedelta(10),
                                     timestep)
        # no more to release
        assert arrays is None

        # reset and try again
        sp.rewind()
        assert sp.num_released == 0
        arrays = sp.release_elements(self.release_time - timedelta(10),
                                     timestep)
        assert arrays is None
        assert sp.num_released == 0

        arrays = sp.release_elements(self.release_time, timestep)
        assert arrays['positions'].shape == (self.num_elements, 3)
        assert np.alltrue(arrays['positions'] == self.start_position)

    def test_cont_release(self):
        sp = PointSourceSurfaceRelease(num_elements=100,
                                start_position=self.start_position,
                                release_time=self.release_time,
                                end_release_time=self.release_time + timedelta(hours=10),
                                )
        timestep = 3600  # one hour in seconds

        # at exactly the release time -- ten get released
        arrays = sp.release_elements(self.release_time, timestep)
        assert arrays['positions'].shape == (10, 3)

        # one hour into release -- ten more released
        arrays = sp.release_elements(self.release_time + timedelta(hours=1),
                                     timestep)
        assert arrays['positions'].shape == (10, 3)
        assert sp.num_released == 20

        # 1-1/2 hours into release - 5 more
        arrays = sp.release_elements(self.release_time + timedelta(hours=2),
                                     timestep / 2)
        assert arrays['positions'].shape == (5, 3)
        assert sp.num_released == 25

        # at end -- rest should be released:
        arrays = sp.release_elements(self.release_time + timedelta(hours=10),
                                     timestep)
        assert arrays['positions'].shape == (75, 3)
        assert sp.num_released == 100

        sp.rewind()

        ## 360 second time step: first LE
        arrays = sp.release_elements(self.release_time, 360)
        assert arrays['positions'].shape == (1, 3)
        assert np.alltrue(arrays['positions'] == self.start_position)

        ## 720 seconds: one more
        arrays = sp.release_elements(self.release_time + timedelta(seconds=360),
                                     360)
        assert arrays['positions'].shape == (1, 3)
        assert np.alltrue(arrays['positions'] == self.start_position)
        assert sp.num_released == 2

    def test_inst_line_release(self):
        # so it's easy to compute where they should be!
        sp = PointSourceSurfaceRelease(num_elements=11,
                                       start_position=(-128.0, 28.0, 0),
                                       release_time=self.release_time,
                                       end_position=(-129.0, 29.0, 0)
                                       )
        timestep = 600  # ten minutes in seconds

        arrays = sp.release_elements(self.release_time, timestep)

        assert arrays['positions'].shape == (11, 3)
        assert np.array_equal(arrays['positions'][:, 0],
                              np.linspace(-128, -129, 11))
        assert np.array_equal(arrays['positions'][:, 1],
                              np.linspace(28, 29, 11))

        assert sp.num_released == 11

    def test_cont_line_release1(self):
        """
        testing a release that is releasing while moving over time

        In this one it all gets released in the first time step.
        """
        # so it's easy to compute where they should be!
        sp = PointSourceSurfaceRelease(num_elements=11,
                                start_position=(-128.0, 28.0, 0),
                                release_time=self.release_time,
                                end_position=(-129.0, 29.0, 0),
                                end_release_time=self.release_time + timedelta(minutes=100)
                                )
        timestep = 100 * 60
        # the full release over one time step
        # (plus a tiny bit to get the last one)
        arrays = sp.release_elements(self.release_time, timestep + 1)

        assert arrays['positions'].shape == (11, 3)
        assert np.array_equal(arrays['positions'][:, 0],
                              np.linspace(-128, -129, 11))
        assert np.array_equal(arrays['positions'][:, 1],
                              np.linspace(28, 29, 11))

        assert sp.num_released == 11

    def test_cont_line_release2(self):
        """
        testing a release that is releasing while moving over time

        first timestep, timestep less than release-time
        """
        sp = PointSourceSurfaceRelease(num_elements=100,
                                start_position=(-128.0, 28.0, 0),
                                release_time=self.release_time,
                                end_position=(-129.0, 29.0, 0),
                                end_release_time=self.release_time + timedelta(minutes=100)
                                )
        lats = np.linspace(-128, -129, 100)
        lons = np.linspace(28, 29, 100)

        # release at release time with time step of 1/10 of release_time
        arrays = sp.release_elements(self.release_time, 10 * 60)

        assert arrays['positions'].shape == (10, 3)
        assert np.array_equal(arrays['positions'][:, 0], lats[:10])
        assert np.array_equal(arrays['positions'][:, 1], lons[:10])
        assert sp.num_released == 10

        # second time step release:
        arrays = sp.release_elements(self.release_time + timedelta(minutes=10),
                                     10 * 60)

        assert arrays['positions'].shape == (10, 3)
        assert np.array_equal(arrays['positions'][:, 0], lats[10:20])
        assert np.array_equal(arrays['positions'][:, 1], lons[10:20])
        assert sp.num_released == 20

    def test_cont_line_release3(self):
        """
        testing a release that is releasing while moving over time

        making sure it's right for the full release
        - multiple elements per step
        """
        sp = PointSourceSurfaceRelease(num_elements=50,
                                start_position=(-128.0, 28.0, 0),
                                release_time=self.release_time,
                                end_position=(-129.0, 30.0, 0),
                                end_release_time=self.release_time + timedelta(minutes=50)
                                )

        #start before release
        time = self.release_time - timedelta(minutes=10)
        delta_t = timedelta(minutes=10)
        timestep = delta_t.total_seconds()
        positions = np.zeros((0, 3), dtype=np.float64)
        # end after release
        while time < self.release_time + timedelta(minutes=100):
            arrays = sp.release_elements(time, timestep)
            if arrays is not None:
                positions = np.r_[positions, arrays['positions']]
            time += delta_t
        assert positions.shape == (50, 3)
        assert np.array_equal(positions[0], (-128.0, 28.0, 0))
        assert np.array_equal(positions[-1], (-129.0, 30.0, 0))
        # check if they are all close to the same line (constant slope)
        diff = np.diff(positions[:, :2], axis=0)
        assert np.alltrue(np.abs(np.diff(diff[:, 0] / diff[:, 1])) < 1e-10)

    def test_cont_line_release4(self):
        """
        testing a release that is releasing while moving over time

        making sure it's right for the full release
        - less than one elements per step
        """
        sp = PointSourceSurfaceRelease(num_elements=10,
                                start_position=(-128.0, 28.0, 0),
                                release_time=self.release_time,
                                end_position=(-129.0, 31.0, 0),
                                end_release_time=self.release_time + timedelta(minutes=50)
                                )

        #start before release
        time = self.release_time - timedelta(minutes=10)
        delta_t = timedelta(minutes=2)
        timestep = delta_t.total_seconds()
        positions = np.zeros((0, 3), dtype=np.float64)
        # end after release
        while time < self.release_time + timedelta(minutes=100):
            arrays = sp.release_elements(time, timestep)
            if arrays is not None:
                positions = np.r_[positions, arrays['positions']]
            time += delta_t
        assert positions.shape == (10, 3)
        assert np.array_equal(positions[0], (-128.0, 28.0, 0))
        assert np.array_equal(positions[-1], (-129.0, 31.0, 0))
        # check for monotonic
        assert np.alltrue(np.sign(np.diff(positions[:, :2], axis=0)) == (-1, 1))

        # check if they are all close to the same line (constant slope)
        diff = np.diff(positions[:, :2], axis=0)
        assert np.alltrue(np.abs(np.diff(diff[:, 0] / diff[:, 1])) < 1e-10)

    positions = [((128.0, 2.0, 0.0), (128.0, -2.0, 0.0)),  # south
                 ((128.0, 2.0, 0.0), (128.0, 4.0, 0.0)),  # north
                 ((128.0, 2.0, 0.0), (125.0, 2.0, 0.0)),  # west
                 ((-128.0, 2.0, 0.0), (-120.0, 2.0, 0.0)),  # east
                 ((-128.0, 2.0, 0.0), (-120.0, 2.01, 0.0)),  # almost east
                 ]

    @pytest.mark.parametrize(('start_position', 'end_position'), positions)
    def test_south_line(self, start_position, end_position):
        """
        testing a line release to the north
        making sure it's right for the full release
        - multiple elements per step
        """
        sp = PointSourceSurfaceRelease(num_elements=50,
                                start_position=start_position,
                                release_time=self.release_time,
                                end_position=end_position,
                                end_release_time=self.release_time + timedelta(minutes=50)
                                )

        #start before release
        time = self.release_time - timedelta(minutes=10)
        delta_t = timedelta(minutes=10)
        timestep = delta_t.total_seconds()
        positions = np.zeros((0, 3), dtype=np.float64)

        # end after release
        while time < self.release_time + timedelta(minutes=100):
            arrays = sp.release_elements(time, timestep)
            if arrays is not None:
                positions = np.r_[positions, arrays['positions']]
            time += delta_t
        assert positions.shape == (50, 3)
        assert np.array_equal(positions[0], start_position)
        assert np.allclose(positions[-1], end_position)
        ##check if they are all close to the same line (constant slope)
        diff = np.diff(positions[:, :2], axis=0)
        if start_position[1] == end_position[1]:
            #horizontal line -- infinite slope
            assert np.alltrue(diff[:, 1] == 0)
        else:
            assert np.alltrue(np.abs(np.diff(diff[:, 0] / diff[:, 1])) < 1e-8)

    def test_cont_not_valid_times(self):
        with pytest.raises(ValueError):
            sp = PointSourceSurfaceRelease(num_elements=100,
                                    start_position=self.start_position,
                                    release_time=self.release_time,
                                    end_release_time=self.release_time - timedelta(seconds=1),
                                    )

    def test_end_position(self):
        """
        if end_position = None, then automatically set it to start_position
        """
        sp = PointSourceSurfaceRelease(num_elements=self.num_elements,
                                start_position=self.start_position,
                                release_time=self.release_time,
                                )

        sp.start_position = (0, 0, 0)
        assert np.any(sp.start_position != sp.end_position)

        sp.end_position = None
        assert np.all(sp.start_position == sp.end_position)

    def test_end_release_time(self):
        """
        if end_release_time = None, then automatically set it to release_time
        """
        sp = PointSourceSurfaceRelease(num_elements=self.num_elements,
                                start_position=self.start_position,
                                release_time=self.release_time,
                                )

        sp.release_time = self.release_time + timedelta(hours=20)
        assert sp.release_time != sp.end_release_time

        sp.end_release_time = None
        assert sp.release_time == sp.end_release_time

num_elements = ((998,),
                (100,),
                (11,),
                (10,),
                (5,),
                (4,),
                (3,),
                (2,),
                )


@pytest.mark.parametrize(('num_elements',), num_elements)
def test_single_line(num_elements):
    """
    various numbers of elemenets over ten time steps, so release
    is less than one, one and more than one per time step.
    """
    print "using num_elements:", num_elements
    start_time = datetime(2012, 1, 1)
    end_time = start_time + timedelta(seconds=100)
    time_step = timedelta(seconds=10)
    start_pos = np.array((0.0, 0.0, 0.0),)
    end_pos = np.array((1.0, 2.0, 0.0),)

    sp = PointSourceSurfaceRelease(num_elements=num_elements,
                             start_position=start_pos,
                             release_time=start_time,
                             end_position=end_pos,
                             end_release_time=end_time,
                             )

    time = start_time
    positions = []
    while time <= end_time + (time_step * 2):
        data = sp.release_elements(time, time_step.total_seconds())
        if data is not None:
            positions.extend(data['positions'])
        time += time_step

    positions = np.array(positions)

    assert len(positions) == num_elements
    assert np.allclose(positions[0], start_pos)
    assert np.allclose(positions[-1], end_pos)
    assert np.allclose(positions[:, 0], np.linspace(start_pos[0],
                                                    end_pos[0],
                                                    num_elements))


def test_line_release_with_one_element():
    """
    one element with a line release
    -- doesn't really make sense, but it shouldn't crash
    """
    start_time = datetime(2012, 1, 1)
    end_time = start_time + timedelta(seconds=100)
    time_step = timedelta(seconds=10)
    start_pos = np.array((0.0, 0.0, 0.0),)
    end_pos = np.array((1.0, 2.0, 0.0),)

    sp = PointSourceSurfaceRelease(num_elements=1,
                             start_position=start_pos,
                             release_time=start_time,
                             end_position=end_pos,
                             end_release_time=end_time,
                             )

    time = start_time - time_step
    assert sp.release_elements(time, time_step.total_seconds()) is None
    time += time_step
    data = sp.release_elements(time, time_step.total_seconds())

    assert np.array_equal(data['positions'], [start_pos, ])


def test_line_release_with_big_timestep():
    """
    a line release: where the timestpe spans before to after the release time
    """
    start_time = datetime(2012, 1, 1)
    end_time = start_time + timedelta(seconds=100)
    time_step = timedelta(seconds=300)
    start_pos = np.array((0.0, 0.0, 0.0),)
    end_pos = np.array((1.0, 2.0, 0.0),)

    sp = PointSourceSurfaceRelease(num_elements=10,
                            start_position=start_pos,
                            release_time=start_time,
                            end_position=end_pos,
                            end_release_time=end_time,
                            )

    data = sp.release_elements(start_time - timedelta(seconds=100),
                               time_step.total_seconds())

    assert np.array_equal(data['positions'][:, 0], np.linspace(0.0, 1.0, 10))
    assert np.array_equal(data['positions'][:, 1], np.linspace(0.0, 2.0, 10))


def test_SpatialRelease():
    """
    see if the right arrays get created
    """
    start_positions = ((0.0, 0.0, 0.0),
                       (28.0, -75.0, 0.0),
                       (-15, 12, 4.0),
                       (80, -80, 100.0),
                       )

    release_time = datetime(2012, 1, 1, 1)
    sp = SpatialRelease(start_positions,
                        release_time,
                        windage_range=(0.01, 0.04),
                        windage_persist=900,
                        )
    data = sp.release_elements(release_time, time_step=600)

    assert data['positions'].shape == (4, 3)


def test_SpatialRelease2():
    """
    make sure they don't release elements twice
    """
    start_positions = ((0.0, 0.0, 0.0),
                       (28.0, -75.0, 0.0),
                       (-15, 12, 4.0),
                       (80, -80, 100.0),
                       )

    release_time = datetime(2012, 1, 1, 1)

    sp = SpatialRelease(start_positions,
                        release_time,
                        windage_range=(0.01, 0.04),
                        windage_persist=900,
                         )

    data = sp.release_elements(release_time, time_step=600)

    assert data['positions'].shape == (4, 3)
    data = sp.release_elements(release_time + timedelta(hours=1), time_step=600)


def test_SpatialRelease3():
    """
    make sure it doesn't release if the first call is too late
    """
    start_positions = ((0.0, 0.0, 0.0),
                       (28.0, -75.0, 0.0),
                       (-15, 12, 4.0),
                       (80, -80, 100.0),
                       )

    release_time = datetime(2012, 1, 1, 1)

    sp = SpatialRelease(start_positions,
                        release_time,
                        windage_range=(0.01, 0.04),
                        windage_persist=900,
                        )
    # first call after release_time
    data = sp.release_elements(release_time + timedelta(seconds=1),
                               time_step=600)
    assert data is None

    # still shouldn't release
    data = sp.release_elements(release_time + timedelta(hours=1),
                               time_step=600)
    assert data is None

    sp.rewind()
    #now it should:
    data = sp.release_elements(release_time, time_step=600)
    assert data['positions'].shape == (4, 3)

def test_PointSourceSurfaceRelease_new_from_dict():
    """
    test to_dict function for Wind object
    create a new wind object and make sure it has same properties
    """
    spill = PointSourceSurfaceRelease(num_elements=1000,
                               start_position=(144.664166, 13.441944, 0.0),
                               release_time=datetime(2013, 2, 13, 9, 0),
                               end_release_time=datetime(2013, 2, 13, 9, 0) + timedelta(hours=6)
                               )

    sp_state = spill.to_dict('create')
    print sp_state

    # this does not catch two objects with same ID
    sp2 = PointSourceSurfaceRelease.new_from_dict(sp_state)

    assert spill == sp2
     
    
def test_PointSourceSurfaceRelease_from_dict():
    """
    test from_dict function for Wind object
    update existing wind object from_dict
    """
    spill = PointSourceSurfaceRelease(num_elements=1000,
                               start_position=(144.664166, 13.441944, 0.0),
                               release_time=datetime(2013, 2, 13, 9, 0),
                               end_release_time=datetime(2013, 2, 13, 9, 0) + timedelta(hours=6)
                               )

    sp_dict = spill.to_dict()
    sp_dict['windage_range'] = [.02, .03]
    spill.from_dict(sp_dict)

    for key in sp_dict.keys():
        if isinstance(spill.__getattribute__(key), np.ndarray):
            np.testing.assert_equal(sp_dict.__getitem__(key),
                                    spill.__getattribute__(key))
        else:
            assert spill.__getattribute__(key) == sp_dict.__getitem__(key)

""" Test OilProps """
def test_OilProps_exceptions():
    from sqlalchemy.orm.exc import NoResultFound
    with pytest.raises(TypeError):
        OilProps(1)
    with pytest.raises(NoResultFound):
        OilProps('test')
        
# just double check values entered correctly
@pytest.mark.parametrize(("oil","density","units"), [('oil_gas',         0.75, 'g/cm^3'),
                                                     ('oil_jetfuels',    0.81, 'g/cm^3'),
                                                     ('oil_4',           0.90, 'g/cm^3'),
                                                     ('oil_crude',       0.90, 'g/cm^3'),
                                                     ('oil_6',           0.99, 'g/cm^3'),
                                                     ('oil_conservative',   1, 'g/cm^3'),
                                                     ('chemical',           1, 'g/cm^3')])
def test_OilProps_sample_oil( oil, density, units):
    """ compare expected values with values stored in OilProps - make sure data entered correctly and unit conversion is correct """
    o = OilProps(oil)
    assert o.get_density(units) == density
    assert o.name == oil

# If DB doesn't exist, it could take awhile to create
# mark this as slow
@pytest.mark.slow
@pytest.mark.parametrize(("oil","api"), [('FUEL OIL NO.6', 12.3)])            
def test_OilProps_DBquery(oil, api):
    """ test dbquery worked for an example like FUEL OIL NO.6 """
    o = OilProps(oil)
    assert o.oil.api == api
    
def test_OilProps_Oil_object():
    """ 
    initialize OilProps from Oil object
    Construction works fine for empty Oil() object. However, get_density() will throw an error
    because Oil().api is undefined for this object. It is the user's responsibility to provide a
    valid (non-empty) Oil object 
    """
    from gnome.db.oil_library.models import Oil
    o = OilProps(Oil()) # this works since we just require an Oil object, but getting
    assert isinstance( o.oil, Oil )
    
    with pytest.raises(ValueError): 
        o.get_density()

if __name__ == "__main__":
    #TC = Test_PointSourceSurfaceRelease()
    #TC.test_model_skips_over_release_time()
    test_SpatialRelease3()
