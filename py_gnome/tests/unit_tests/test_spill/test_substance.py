from pathlib import Path

from gnome.spill.substance import (Substance,
                                   GnomeOil,
                                   NonWeatheringSubstance)

from gnome.spill.initializers import InitWindages

try:
    import adios_db
    ADIOS_IS_THERE = True
    del adios_db
except ImportError:
    ADIOS_IS_THERE = False

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


class TestGnomeOil(object):

    def test_init(self):
        oil1 = GnomeOil('oil_ans_mp', windage_range=(0.05, 0.07))
        assert oil1.windage_range == (0.05, 0.07)
        # assert isinstance(oil1.record, Oil)	#GnomeOil doesn't have a record
        assert isinstance(oil1.initializers[0], InitWindages)
        initw = oil1.initializers[0]
        assert all([atype in oil1.array_types for atype in initw.array_types])

    @pytest.mark.skipif(not ADIOS_IS_THERE, reason="requires the adios_db package")
    def test_oil_from_adios_db_json(self):

        go = GnomeOil(name="dummy name",
                            filename=str(DATA_DIR / "ANS_EC02713.json"))

        print(go.name)

        ## fixme: this is NOT the name it should get!
        assert go.name == "dummy name"



    def test_eq(self):
        sub1 = GnomeOil('oil_ans_mp')
        sub2 = GnomeOil('oil_ans_mp')
        assert sub1 == sub2
        sub3 = GnomeOil('oil_ans_mp', windage_range=(0.05, 0.07))
        assert sub1 != sub3

    def test_hashable_1(self):
        """
        GnomeOIl needs to be hashable, so that is can be used with lru_cache

        This only tests that the SAME oil object is hashable and recoverable,
        but that's OK for caching.

        NOTE: This doesn't test at all whether the hash "works"
              e.g. that two different GnomeOils don't hash the same
              That may be OK -- as equality works, yes?
        """
        oil1 = GnomeOil('oil_ans_mp')

        # can I put it in a dict, and get it out again?
        d = {}
        d[oil1] = "yes"

        assert d[oil1] == "yes"

    def test_serialization(self):
        oil1 = GnomeOil('oil_ans_mp', windage_range=(0.05, 0.07))
        ser = oil1.serialize()
        deser = GnomeOil.deserialize(ser)
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
        test_obj = GnomeOil('oil_ans_mp', windage_range=(0.05, 0.07))
        json_, savefile, refs = test_obj.save(saveloc_)
        test_obj2 = test_obj.__class__.load(savefile)
        assert test_obj == test_obj2

    def test_set_emulsification_constants(self):
        test_obj = GnomeOil('oil_ans_mp')
        assert test_obj._bulltime is None
        assert test_obj.bullwinkle < 0.5
        # assert test_obj.bullwinkle < 0.5 and test_obj.bullwinkle is test_obj.record.bullwinkle_fraction
        d = test_obj.serialize()
        d['bullwinkle_time'] = 60
        d['bullwinkle_fraction'] = 0.7
        test_obj.update_from_dict(d)
        assert test_obj.bullwinkle == 0.7
        assert test_obj.serialize()['bullwinkle_fraction'] == 0.7
        assert test_obj.bulltime == 60
        assert test_obj.serialize()['bullwinkle_time'] == 60

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