from datetime import datetime, timedelta

from gnome.spill.spill import (Spill)
from gnome.spill.substance import NonWeatheringSubstance
from gnome.spill.release import PointLineRelease

import pytest

@pytest.fixture('function')
def sp():
    return Spill()


class TestSpill(object):
    rel_time = datetime(2014, 1, 1, 0, 0)
    pos = (0, 1, 2)

    def test__init(self):
        sp = Spill()
        #assert default construction
        assert sp.substance and isinstance(sp.substance, NonWeatheringSubstance)
        assert sp.release  and isinstance(sp.release, PointLineRelease)

    def test_num_per_timestep_release_elements(self):
        'release elements in the context of a spill container'
        # todo: need a test for a line release where rate is given - to check
        # positions are being initialized correctly
        end_time = self.rel_time + timedelta(hours=1)
        release = PointLineRelease(self.rel_time,
                                   self.pos,
                                   num_elements=1000,
                                   end_release_time=end_time)
        sp = Spill(release=release)
        sp.prepare_for_model_run({}, 900)
        for ix in range(5):
            model_time = self.rel_time + timedelta(seconds=900 * ix)
            to_rel = sp.release_elements(model_time, 900)
            if model_time < sp.end_release_time:
                assert to_rel == 250
                assert sp.num_released == min((ix+1) * 250, 1000)
            else:
                assert to_rel == 0

    def test_amount(self, sp):
        assert sp.amount == 0
        assert sp.release.release_mass == 0
        sp.units = 'm^3'
        sp.amount = 10
        assert sp.release.release_mass == 10 * sp.substance.standard_density
        assert sp.amount == 10

        with pytest.raises(ValueError):
            sp.amount = -1

    def test_units(self, sp):
        assert sp.units == 'kg'
        sp.units = 'lb'
        assert sp.units == 'lb'
        sp.units = 'gal'
        assert sp.units == 'gal'
        with pytest.raises(ValueError):
            sp.units = 'inches'

    def test_prepare_for_model_run(self, sp):
        sp.prepare_for_model_run({}, 1)
