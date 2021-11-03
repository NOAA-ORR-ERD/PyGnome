from pathlib import Path
from gnome.spills.substance import (Substance,
                                   NonWeatheringSubstance)

from gnome.spills.initializers import InitWindages

import pytest

DATA_DIR = Path(__file__).parent / "data_for_tests"


class TestSubstance:
    '''Test for base class'''

    def test_init(self):
        # check default state
        sub1 = Substance()
        assert len(sub1.initializers) == 1
        assert isinstance(sub1.initializers[0], InitWindages)
        initw = sub1.initializers[0]
        # substance should expose the array types of it's initializers
        assert all([atype in sub1.array_types for atype in initw.array_types])
        # in this case, there should only be InitWindages
        assert list(sub1.array_types.keys()) == list(initw.array_types.keys())

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