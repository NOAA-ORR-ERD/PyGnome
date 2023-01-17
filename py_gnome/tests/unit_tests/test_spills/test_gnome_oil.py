from pathlib import Path
from math import isclose
from copy import copy

from gnome.spills.gnome_oil import GnomeOil

from gnome.spills import sample_oils

from ..conftest import (test_oil)

try:
    import adios_db
    ADIOS_IS_THERE = True
    del adios_db
except ImportError:
    ADIOS_IS_THERE = False

import pytest

DATA_DIR = Path(__file__).parent / "data_for_tests"


class TestGnomeOil:

    def test_init_sample_oil(self):
        oil1 = GnomeOil('oil_ans_mp',
                        windage_range=(0.05, 0.07))
        assert oil1.windage_range == (0.05, 0.07)

    def test_init_from_oil_dict(self):
        #breakpoint()
        go = GnomeOil(**sample_oils.oil_bahia)

        assert go.pour_point == 310.9278

    def test_init_from_oil_dict_and_bad_filename(self):
        kwargs = copy(sample_oils.oil_bahia)
        kwargs['filename'] = str(DATA_DIR / "bogus.json")

        go = GnomeOil(**kwargs)

        assert go.pour_point == 310.9278

    def test_init_from_oil_dict_and_good_filename(self):
        kwargs = copy(sample_oils.oil_bahia)
        kwargs['filename'] = str(DATA_DIR / "ANS_EC02713.json")

        go = GnomeOil(**kwargs)

        assert go.pour_point == 310.9278

    def test_deserialize_from_good_filename(self):
        substance_json = {
            'obj_type': 'gnome.spills.gnome_oil.GnomeOil',
            'filename': str(DATA_DIR / "ANS_EC02713.json"),
            'name': "Alaska North Slope [2015]"
        }

        go = GnomeOil.deserialize(substance_json)

        assert isclose(go.pour_point, 222.15)

    # This check has been disabled -- breaking the copy for uncertainty
    # def test_pass_both(self):
    #     with pytest.raises(TypeError):
    #         oil = GnomeOil(oil_name = 'oil_ans_mp',
    #                        filename=str(DATA_DIR / "ANS_EC02713.json"))

    @pytest.mark.skipif(not ADIOS_IS_THERE, reason="requires the adios_db package")
    def test_oil_from_adios_db_json(self):

        go = GnomeOil(filename=str(DATA_DIR / "ANS_EC02713.json"))

        # maybe there should be more tests here?
        # or not -- the make_gnome_oil code should be tested elsewhere.
        assert go.name == "Alaska North Slope [2015]"

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

    def test_invalid_oil_name(self):
        with pytest.raises(ValueError):
            oil = GnomeOil('non-existant-oil')

    def test_invalid_file_name(self):
        with pytest.raises(FileNotFoundError) as err:
            oil = GnomeOil(filename='non-existant-oil')

    def test_serialization(self):
        oil1 = GnomeOil('oil_ans_mp', windage_range=(0.05, 0.07))
        ser = oil1.serialize()
        deser = GnomeOil.deserialize(ser)

        assert deser == oil1
        assert deser.initializers[0].windage_range == oil1.windage_range
        assert isclose(deser.standard_density, oil1.standard_density, rel_tol=1e-5)

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

        print(f"{test_obj._diff(test_obj2)}")
        assert test_obj == test_obj2

    def test_set_emulsification_constants(self):
        test_obj = GnomeOil('oil_ans_mp')

        assert test_obj.bullwinkle_time == -999.0
        assert test_obj.bullwinkle_fraction < 0.5

        d = test_obj.serialize()
        d['bullwinkle_time'] = 60
        d['bullwinkle_fraction'] = 0.7
        test_obj.update_from_dict(d)

        assert test_obj.bullwinkle_fraction == 0.7
        assert test_obj.serialize()['bullwinkle_fraction'] == 0.7
        assert test_obj.bullwinkle_time == 60
        assert test_obj.serialize()['bullwinkle_time'] == 60

    def test_bulltime(self):
        '''
        user set time to start emulsification
        This should be in the GnomeOil tests
        '''
        oil = GnomeOil(test_oil)
        assert oil.bullwinkle_time == -999

        oil.bullwinkle_time = 3600
        assert oil.bullwinkle_time == 3600


    def test_bullwinkle(self):
        '''
        user set emulsion constant
        This should be in the GnomeOil tests ...
        '''

        oil = GnomeOil(test_oil)

        # our test_oil is the sample oils
        assert isclose(oil.bullwinkle_fraction, 0.1937235)

        oil.bullwinkle_fraction = .4
        assert oil.bullwinkle_fraction == .4
