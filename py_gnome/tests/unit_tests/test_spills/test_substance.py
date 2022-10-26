from pathlib import Path
from gnome.spills.substance import (Substance,
                                    NonWeatheringSubstance,
                                    SubsurfaceSubstance)

from gnome.spills.initializers import InitWindages

import pytest

DATA_DIR = Path(__file__).parent / "data_for_tests"

# fixme: There should be tests of the substance "model" API here
# what does a Substance do? is that working?


class TestSubstance:
    '''Test for base class'''

    def test_init(self):
        # check default state
        sub1 = Substance()
        assert isinstance(sub1._windage_init, InitWindages)
        expected_arrays = {'windages', 'windage_range', 'windage_persist', 'density', 'fate_status'}
        assert sub1.array_types.keys() == expected_arrays
        assert sub1.windage_range == (0.01, 0.04)
        assert sub1.windage_persist == 900

    def test_init_inf_persist(self):
        """
        setting windage_persist to inf should set it to -1
        """
        sub1 = Substance(windage_persist=float("Inf"))
        assert sub1.windage_persist == -1

    def test_inf_persist(self):
        """
        setting windage_persist to inf should set it to -1
        """
        sub1 = Substance()
        sub1.windage_persist = float("Inf")
        assert sub1.windage_persist == -1

    def test_setters(self):
        """
        setting windage values after should "take"
        """
        sub1 = Substance()
        sub1.windage_persist = 500
        sub1.windage_range = (0.0, 0.03)

        assert sub1.windage_persist == 500
        assert sub1.windage_range == (0.0, 0.03)

    def test_eq(self):
        sub1 = Substance()
        sub2 = Substance()
        assert sub1 == sub2
        sub3 = Substance(windage_range=(0.05, 0.07))
        assert sub1 != sub3

    def test_serialization(self):
        sub1 = Substance(windage_range=(0.05, 0.07))
        ser = sub1.serialize()
        deser = Substance.deserialize(ser)
        assert deser == sub1
        assert deser.initializers[0].windage_range == sub1.windage_range

    def test_save_load(self, saveloc_):
        '''
        test save/load for initializers and for ElementType objects containing
        each initializer. Tests serialize/deserialize as well.
        These are stored as nested objects in the Spill but this should also work
        so test it here
        '''
        test_obj = Substance(windage_range=(0.05, 0.07))
        json_, savefile, refs = test_obj.save(saveloc_)
        test_obj2 = test_obj.__class__.load(savefile)
        assert test_obj == test_obj2


class TestNonWeatheringSubstance(object):

    def test_init(self):
        nws1 = NonWeatheringSubstance(windage_range=(0.05, 0.07))
        assert nws1.windage_range == (0.05, 0.07)
        assert isinstance(nws1.initializers[0], InitWindages)
        initw = nws1.initializers[0]
        assert all([atype in nws1.array_types for atype in initw.array_types])

    def test_eq(self):
        sub1 = NonWeatheringSubstance()
        sub2 = NonWeatheringSubstance()
        assert sub1 == sub2
        sub3 = NonWeatheringSubstance(windage_range=(0.05, 0.07))
        assert sub1 != sub3

    def test_serialization(self):
        oil1 = NonWeatheringSubstance(windage_range=(0.05, 0.07))
        ser = oil1.serialize()
        deser = NonWeatheringSubstance.deserialize(ser)
        assert deser == oil1
        assert deser.initializers[0].windage_range == oil1.windage_range
        assert deser.standard_density == oil1.standard_density

    def test_save_load(self, saveloc_):
        '''
        test save/load for initializers and for ElementType objects containing
        each initializer. Tests serialize/deserialize as well.
        These are stored as nested objects in the Spill but this should also work
        so test it here
        '''
        test_obj = NonWeatheringSubstance(windage_range=(0.05, 0.07))
        json_, savefile, refs = test_obj.save(saveloc_)
        test_obj2 = test_obj.__class__.load(savefile)
        assert test_obj == test_obj2


class Test_SubsurfaceSubstance:
    """
    Tests of Substance that support rise velocity

    Needed for vertical movers :-)

    Fixme: this should test the initilization!
    """

    @pytest.mark.parametrize('distribution', ['UniformDistribution',
                                              'NormalDistribution',
                                              'LogNormalDistribution',
                                              'WeibullDistribution',
                                              ])
    def test_init_with_distribution(self, distribution):
        """
        NOTE: this only tests whether the init fails,
        not whether it does the right thing
        """
        subs = SubsurfaceSubstance(distribution=distribution)

        print(subs.array_types.keys())
        expected_arrays = {'rise_vel',
                           'droplet_diameter',
                           'windages',
                           'windage_range',
                           'windage_persist',
                           'density',
                           'fate_status'}

        assert subs.all_array_types.keys() == expected_arrays



