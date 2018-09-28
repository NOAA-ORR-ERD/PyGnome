from datetime import datetime, timedelta

from gnome.spill.spill import (BaseSpill,
                               Spill)
from gnome.spill.substance import NonWeatheringSubstance
from gnome.spill.release import ContinuousRelease


class TestBaseSpill(object):

    def test_init(self):
        bs = BaseSpill()
        assert isinstance(bs.substance, NonWeatheringSubstance)

class TestSpill(object):
    rel_time = datetime(2014, 1, 1, 0, 0)
    pos = (0, 1, 2)

    def test__init(self):
        sp = Spill()
        assert isinstance(sp.substance, NonWeatheringSubstance)

    def test_num_per_timestep_release_elements(self):
        'release elements in the context of a spill container'
        # todo: need a test for a line release where rate is given - to check
        # positions are being initialized correctly
        end_time = self.rel_time + timedelta(hours=1)
        release = ContinuousRelease(self.rel_time,
                                    self.pos,
                                    num_per_timestep=100,
                                    initial_elements=1000,
                                    end_release_time=end_time)
        sp = Spill(release=release)
        sp.prepare_for_model_run({})
        for ix in range(5):
            model_time = self.rel_time + timedelta(seconds=900 * ix)
            num_les = sp.release_elements(model_time, 900)
            if model_time <= sp.end_release_time:
                if ix == 0:
                    assert num_les == 1100
                else:
                    assert num_les == 100
                assert sp.num_released == 100 + ix * 100 + 1000
            else:
                assert num_les == 0
