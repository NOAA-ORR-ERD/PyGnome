'''
tests for cleanup options
'''
from datetime import datetime, timedelta

from pytest import mark
import numpy as np

from gnome.basic_types import oil_status
from gnome.environment import Water
from gnome.weatherers.intrinsic import IntrinsicProps

from gnome.weatherers import Skimmer, Burn
from gnome.spill_container import SpillContainer
from gnome.spill import point_line_release_spill
from gnome.utilities.inf_datetime import InfDateTime

from ..conftest import test_oil

delay = 1.
time_step = 900

rel_time = datetime(2014, 1, 1, 0, 0)
active_start = rel_time + timedelta(seconds=time_step)
active_stop = active_start + timedelta(hours=1.)
amount = 36000.
units = 'kg'    # leave as SI units


def test_objs():
    '''
    function for created tests SpillContainer and test IntrinsicProps object
    test objects so we can run Skimmer, Burn like a model without using a
    full on Model
    '''
    intrinsic = IntrinsicProps(Water())
    sc = SpillContainer()
    sc.spills += point_line_release_spill(10,
                                          (0, 0, 0),
                                          rel_time,
                                          substance=test_oil,
                                          amount=amount,
                                          units='kg')
    return (sc, intrinsic)


class TestSkimmer:
    skimmer = Skimmer(amount,
                      'kg',
                      efficiency=0.3,
                      active_start=active_start,
                      active_stop=active_stop)

    (sc, intrinsic) = test_objs()

    def reset_test_objs(self):
        '''
        reset the test objects
        '''
        self.sc.rewind()
        self.sc.prepare_for_model_run(self.intrinsic.array_types)
        self.intrinsic.initialize(self.sc)

    def reset_and_release(self):
        '''
        reset test objects and relaese elements
        '''
        self.reset_test_objs()
        num_rel = self.sc.release_elements(time_step, rel_time)
        self.intrinsic.update(num_rel, self.sc)

    def test_prepare_for_model_run(self):
        self.reset_and_release()
        self.skimmer.prepare_for_model_run(self.sc)
        assert self.sc.weathering_data['skimmed'] == 0.
        assert (self.skimmer._rate ==
                self.skimmer.amount/(self.skimmer.active_stop -
                                     self.skimmer.active_start).total_seconds())

    @mark.parametrize(("model_time", "active", "ts"),
                      [(active_start, True, time_step),
                       (active_start - timedelta(seconds=time_step/3.), True,
                        time_step*2./3.),  # between active_start - active_stop
                       (active_start + timedelta(seconds=time_step/2.), True,
                        time_step),     # between active_start - active_stop
                       (active_stop, False, None),
                       (active_stop - timedelta(seconds=time_step), True,
                        time_step),     # before active stop
                       (active_stop - timedelta(seconds=time_step/3.), True,
                        time_step/3.)   # before active stop
                       ])
    def test_prepare_for_model_step(self, model_time, active, ts):
        '''
        assert that the _timestep property is being set correctly. This is
        less than or equal to time_step specified by model. Since we define
        a rate object, this is just to ensure the amount Skimmed between the
        duration specified matches given a constant skim rate.
        '''
        self.reset_and_release()
        self.skimmer.prepare_for_model_step(self.sc, time_step, model_time)

        assert self.skimmer.active is active
        if active:
            assert self.skimmer._timestep == ts

    def test_weather_elements(self):
        '''
        Test only the mass removed due to Skimmer operation:
        1) sc['mass'] + sc.weathering_data['skimmed'] =  spill_amount
        2) sc.weathering_data['skimmed']/skimmer.amount = skimmer.efficiency
        '''
        self.reset_test_objs()
        self.skimmer.prepare_for_model_run(self.sc)

        assert self.sc.weathering_data['skimmed'] == 0.0

        model_time = rel_time
        while (model_time <
               self.skimmer.active_stop + timedelta(seconds=2*time_step)):

            amt_skimmed = self.sc.weathering_data['skimmed']

            num_rel = self.sc.release_elements(time_step, model_time)
            self.intrinsic.update(num_rel, self.sc)
            self.skimmer.prepare_for_model_step(self.sc, time_step, model_time)

            self.skimmer.weather_elements(self.sc, time_step, model_time)

            if not self.skimmer.active:
                assert self.sc.weathering_data['skimmed'] == amt_skimmed
            else:
                # check total amount removed at each timestep
                assert self.sc.weathering_data['skimmed'] > amt_skimmed

            self.skimmer.model_step_is_done(self.sc)
            self.sc.model_step_is_done()
            # model would do the following
            self.sc['age'][:] = self.sc['age'][:] + time_step
            model_time += timedelta(seconds=time_step)

            # check - useful for debugging issues with recursion
            assert (amount == self.sc.weathering_data['skimmed'] +
                    self.sc['mass'].sum())

        # following should finally hold true for entire run
        assert np.allclose(amount, self.sc.weathering_data['skimmed'] +
                           self.sc['mass'].sum(), atol=1e-6)
        assert np.allclose(self.sc.weathering_data['skimmed']/self.skimmer.amount,
                           self.skimmer.efficiency, atol=1e-6)


class TestBurn:
    (sc, intrinsic) = test_objs()
    spill = sc.spills[0]
    op = spill.get('substance')
    volume = spill.get_mass()/op.get_density(intrinsic.water.temperature)
    thick = 1
    area = (0.5 * volume)/thick
    burn = Burn(area, thick, active_start)

    def reset_test_objs(self):
        '''
        reset test objects
        '''
        self.sc.rewind()
        at = self.intrinsic.array_types
        at.update(self.burn.array_types)
        self.intrinsic.initialize(self.sc)
        self.sc.prepare_for_model_run(at)

    def test_init(self):
        '''
        active_stop is ignored if set by user
        '''
        burn = Burn(self.area,
                    self.thick,
                    active_start=active_start,
                    active_stop=active_start,
                    name='test_burn',
                    on=False)   # this is ignored!
        assert burn.active_stop == InfDateTime('inf')
        assert burn.name == 'test_burn'
        assert not burn.on

    def test_prepare_for_model_run(self):
        ''' check _curr_thickness is reset'''
        self.burn._curr_thickness = 0.002
        self.burn.prepare_for_model_run(self.sc)
        assert self.burn._curr_thickness == self.burn.thickness

    def test_prepare_for_model_step(self):
        '''
        once thickness reaches _min_thickness, make the mover inactive
        '''
        burn = Burn(self.area,
                    self.thick,
                    active_start=active_start)
        burn.prepare_for_model_run(self.sc)
        self._curr_thickness = 0.002
        burn.prepare_for_model_step(self.sc, time_step, rel_time)
        assert not burn._active

        # make it inactive
        burn.on = False
        burn.prepare_for_model_step(self.sc, time_step, rel_time)
        assert not burn._active

    @mark.parametrize(("thick", "frac_water"), [(0.003, None),
                                                (1, None),
                                                (1, 0.5)])
    def test_weather_elements_zero_frac_water(self, thick, frac_water):
        '''
        weather elements and test. frac_water is 0

        1) tests the expected burned mass equals 'burned' amount stored in
           weathering_data
        2) also tests the mass_remaining is consistent with what we expect

        Also sets the 'frac_water' to 0.5 for one of the tests just to ensure
        it works. No assertion for testing that water_frac > 0 runs longer but
        this is visually checked for this example.
        '''
        self.spill.set('num_elements', 500)
        self.reset_test_objs()
        area = (0.5 * self.volume)/thick
        burn = Burn(area, thick, active_start)

        model_time = rel_time
        burn.prepare_for_model_run(self.sc)

        # once burn becomes active, sc.weathering_data['burned'] > 0.0 and
        # burn becomes inactive once
        # burn._curr_thickness == burn._min_thickness
        while burn.active or self.sc.weathering_data['burned'] == 0.0:
            step_num = (model_time - rel_time).total_seconds() // time_step
            num = self.sc.release_elements(time_step, model_time)
            if num > 0 and frac_water is not None:
                self.sc['frac_water'][:] = frac_water
            self.intrinsic.update(num, self.sc)

            dt = timedelta(seconds=time_step)
            burn.prepare_for_model_step(self.sc, time_step, model_time)

            if burn._curr_thickness <= burn._min_thickness:
                # inactive flag is set in prepare_for_model_step() - not set in
                # weather_elements()
                assert not burn._active
            elif model_time + dt <= burn.active_start:
                # if model_time + dt == burn.active_start, then start the burn
                # in next step
                assert not burn._active
            else:
                assert burn._active

            burn.weather_elements(self.sc, time_step, model_time)

            self.sc['age'][:] = self.sc['age'][:] + time_step
            model_time += dt

            if step_num > 100:
                '''
                just break it after some threshold, though should not be needed
                '''
                assert False
                break

        # following should finally hold true for entire run
        assert np.allclose(amount, self.sc.weathering_data['burned'] +
                           self.sc['mass'].sum(), atol=1e-6)
        assert burn._burn_duration > 0.0

        # since frac_water is zero, expected burn is known
        exp_burned = ((burn.thickness - burn._curr_thickness) * burn.area *
                      self.op.get_density())
        assert np.isclose(self.sc.weathering_data['burned'], exp_burned)

        # given LEs are discrete elements, we cannot add a fraction of an LE
        # to be burned. Thus, the LEs marked to be burnt will have a mass >=
        # mass specified from area * thickness * density(). For the assertion
        # below include this extra mass.
        mass_per_le = self.sc['init_mass'][0]
        extra_init_mass = mass_per_le - (burn.thickness * burn.area *
                                         self.op.get_density() % mass_per_le)
        exp_mass_remain = (burn._curr_thickness * burn.area *
                           self.op.get_density() + extra_init_mass)
        mass_remain_for_burn_LEs = self.sc['mass'][self.sc['status_codes'] ==
                                                   oil_status.burn].sum()
        assert np.allclose(exp_mass_remain, mass_remain_for_burn_LEs)

        print '\nCompleted steps: {0:2}'.format(step_num)
        print ('Current Thickness: {0:.3f}, '
               'Duration (hrs): {1:.3f}').format(burn._curr_thickness,
                                                 burn._burn_duration/3600)
