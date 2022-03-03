
from datetime import datetime, timedelta

from gnome.spill.spill import (Spill)
from gnome.spill.substance import NonWeatheringSubstance
from gnome.spill.release import PointLineRelease
from gnome.spill_container import SpillContainer

import pytest


@pytest.fixture(scope='function')
def sp():
    return Spill()


rel_time = datetime(2014, 1, 1, 0, 0)
end_rel_time = rel_time + timedelta(seconds=9000)
pos = (0, 1, 2)
end_release_pos = (1, 2, 3)


@pytest.fixture(scope='function')
def inst_point_spill():
    release = PointLineRelease(rel_time, pos)
    return Spill(release=release,
                 amount=5000)


@pytest.fixture(scope='function')
def inst_point_line_spill():
    release = PointLineRelease(rel_time,
                               pos,
                               end_position=end_release_pos)
    return Spill(release=release,
                 amount=5000)


@pytest.fixture(scope='function')
def cont_point_spill():
    release = PointLineRelease(rel_time,
                               pos,
                               end_release_time=end_rel_time)
    return Spill(release=release,
                 amount=5000)


@pytest.fixture(scope='function')
def cont_point_line_spill():
    release = PointLineRelease(rel_time,
                               pos,
                               end_position=end_release_pos,
                               end_release_time=end_rel_time)
    return Spill(release=release,
                 amount=5000)


@pytest.fixture(scope='function')
def cont_point_spill_le_per_ts():
    release = PointLineRelease(rel_time,
                               pos,
                               end_release_time=end_rel_time,
                               num_per_timestep=200)
    return Spill(release=release,
                 amount=5000)


class TestSpill:
    rel_time = datetime(2014, 1, 1, 0, 0)
    pos = (0, 1, 2)

    def test__init(self):
        sp = Spill()
        assert sp.substance and isinstance(sp.substance, NonWeatheringSubstance)
        assert sp.release and isinstance(sp.release, PointLineRelease)


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
        sc = SpillContainer()
        sc.spills += sp
        sc.prepare_for_model_run(array_types=sp.array_types)
        sp.prepare_for_model_run(900)
        for ix in range(5):
            model_time = self.rel_time + timedelta(seconds=900 * ix)
            to_rel = sp.release_elements(sc, model_time, 900)
            if model_time < sp.end_release_time:
                assert to_rel == 250
                assert len(sc['spill_num']) == min((ix + 1) * 250, 1000)
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

    # These are for when SpillContainer is removed
    # NOTE: you can not parametrize on fixtures like this
    # how to parametrize on fixtures:
    # https://miguendes.me/how-to-use-fixtures-as-arguments-in-pytestmarkparametrize
    # klunky, but it seems to work
    # it uses: request.getfixturevalue
    @pytest.mark.xfail()  ## these tests need some refactoring ...
    @pytest.mark.parametrize('spill', ["inst_point_spill",
                                       # "inst_point_line_spill",
                                       # "cont_point_spill",
                                       # "cont_point_line_spill",
                                       # "cont_point_spill_le_per_ts",
                                       ])
    def test_spill_behavior(self, spill, request):
        '''
        Validates spill behavior for a number of example spills.

        Tests the following items by simulating model runs:
        1. Over a timestep-aligned model run:
            a. elements are not releaed before spill begins
            b. expected number of elements exist at any timestep
            c. expected number of elements exist at end of spill
            d. LEs are initialized to expected mass and position values
        2. Rewind resets appropriate fields, LEData, and release
        3. Over a non-aligned model run:
            a. correct fraction of LEs are released on overlapping start & end
            b. expected number of elements exist at any timestep
            c. expected number of elements exist at end of spill
            d. LEs are initialized to expected mass and position values
            e. maximum mass error does not exceed 1 LE at any time.
        '''
        spill = request.getfixturevalue(spill)
        print(f"{type(spill)=}, {spill=}")

        sc = SpillContainer()
        sc.array_types.update(spill.all_array_types)
        print(f"{sc.array_types=}")

        sc.spills += spill

        ts = 900
        tsd = timedelta(seconds=ts)
        model_time = spill.release_time - tsd

        spill.prepare_for_model_run(900)
        le_per_ts = spill.release.LE_timestep_ratio(900)
        mass_per_le = spill.release._mass_per_le
        for ix in range(0, 20):
            new_rel = spill.release_elements(sc, model_time, ts)
            if ix == 0:
                assert spill._num_released == 0
            elif model_time < spill.end_release_time:
                assert spill.num_released == le_per_ts * ix
                assert sum(spill.data['mass']) == le_per_ts * mass_per_le * ix
                assert all(spill.data['density'] == spill.substance.standard_density)
            else:
                if spill.release.num_elements:
                    assert spill.num_released == spill.release.num_elements
                else:
                    assert spill.num_released == spill.release.num_per_timestep * spill.release.get_num_release_time_steps(900)
                assert sum(spill.data['mass']) == spill.release.release_mass
            model_time += tsd
        spill.rewind()
        assert spill.num_released == 0
        assert spill.release._prepared == False

        model_time = spill.release_time - timedelta(seconds=ts*4/3)
        spill.prepare_for_model_run(900)
        le_per_ts = spill.release.LE_timestep_ratio(900)
        mass_per_le = spill.release._mass_per_le
        for ix in range(0,20):
            new_rel = spill.release_elements(model_time, ts)
            if ix == 0:
                assert spill.num_released == 0
            elif model_time + tsd < spill.end_release_time and ix == 1:
                assert spill.num_released == int(0.667 * le_per_ts)
                assert sum(spill.data['mass']) == int(0.667 * le_per_ts) * mass_per_le
                assert (0.667 * le_per_ts * mass_per_le) - sum(spill.data['mass']) < mass_per_le
            elif model_time + tsd < spill.end_release_time:
                assert spill.num_released % 100 == 66
            else:
                assert new_rel % 2 == 0 #check ensures last LE was actually released
                if spill.release.num_elements:
                    assert spill.num_released == spill.release.num_elements
                else:
                    assert spill.num_released == spill.release.num_per_timestep * spill.release.get_num_release_time_steps(900)
                assert sum(spill.data['mass']) == spill.release.release_mass
            model_time += tsd


