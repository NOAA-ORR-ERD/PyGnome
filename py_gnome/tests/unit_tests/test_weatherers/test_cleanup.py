'''
tests for cleanup options
'''
from datetime import datetime, timedelta

from pytest import mark
import numpy as np

from gnome.basic_types import oil_status
from gnome.environment import Water
from gnome.weatherers.intrinsic import WeatheringData

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
    function for created tests SpillContainer and test WeatheringData object
    test objects so we can run Skimmer, Burn like a model without using a
    full on Model
    '''
    intrinsic = WeatheringData(Water())
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
            mask = self.sc['status_codes'] == oil_status.skim
            assert mask.sum() > 0

    @mark.parametrize("avg_frac_water", [0.0, 0.4])
    def test_weather_elements(self, avg_frac_water):
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
            if num_rel > 0:
                self.sc['frac_water'][:] = avg_frac_water
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
        # efficiency decreased since only (1 - avg_frac_water) is the fraction
        # of oil collected by skimmer
        assert np.allclose(self.sc.weathering_data['skimmed']/self.skimmer.amount,
                           self.skimmer.efficiency * (1 - avg_frac_water),
                           atol=1e-6)


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
        ''' check _oil_thickness is reset'''
        self.burn._oil_thickness = 0.002
        self.burn.prepare_for_model_run(self.sc)
        assert self.burn._oil_thickness is None

    def test_prepare_for_model_step(self):
        '''
        once thickness reaches _min_thickness, test the mover becomes inactive
        if mover is off, it is also inactive
        '''
        burn = Burn(self.area,
                    self.thick,
                    active_start=active_start)
        burn.prepare_for_model_run(self.sc)
        self._oil_thickness = 0.002
        burn.prepare_for_model_step(self.sc, time_step, rel_time)
        assert not burn._active

        # turn it off
        burn.on = False
        burn.prepare_for_model_step(self.sc, time_step, rel_time)
        assert not burn._active

    def _weather_elements_helper(self, burn, avg_frac_water=0.0):
        '''
        refactored model run from test_weather_elements to this helper function
        It is also used by next test:
        test_elements_weather_slower_with_frac_water
        '''
        self.reset_test_objs()

        model_time = rel_time
        burn.prepare_for_model_run(self.sc)

        # once burn becomes active, sc.weathering_data['burned'] > 0.0 and
        # burn becomes inactive once
        # burn._oil_thickness == burn._min_thickness
        init_oil_thickness = None
        while burn.active or self.sc.weathering_data['burned'] == 0.0:
            step_num = (model_time - rel_time).total_seconds() // time_step
            num = self.sc.release_elements(time_step, model_time)
            if num > 0:
                self.sc['frac_water'][:] = avg_frac_water
            self.intrinsic.update(num, self.sc)

            dt = timedelta(seconds=time_step)
            burn.prepare_for_model_step(self.sc, time_step, model_time)

            if burn._oil_thickness <= burn._min_thickness:
                # inactive flag is set in prepare_for_model_step() - not set in
                # weather_elements()
                assert not burn._active
            elif model_time + dt <= burn.active_start:
                # if model_time + dt == burn.active_start, then start the burn
                # in next step
                assert not burn._active
            else:
                if init_oil_thickness is None:
                    init_oil_thickness = burn._oil_thickness

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

        assert burn._burn_duration > 0.0
        print '\nCompleted steps: {0:2}'.format(step_num)

        return init_oil_thickness

    @mark.parametrize(("thick", "avg_frac_water"), [(0.003, 0),
                                                    (1, 0),
                                                    (1, 0.3)])
    def test_weather_elements(self, thick, avg_frac_water):
        '''
        weather elements and test. frac_water is 0

        1) tests the expected burned mass equals 'burned' amount stored in
           weathering_data
        2) also tests the mass_remaining is consistent with what we expect
        3) tests the mass of LEs set for burn equals the mass of oil given
           avg_frac_water and the thickness, and area. Since we cannot have
           a fraction of an LE, the difference should be within the mass of
           one LE.

        Also sets the 'frac_water' to 0.5 for one of the tests just to ensure
        it works.
        '''
        self.spill.set('num_elements', 500)
        area = (0.5 * self.volume)/thick
        burn = Burn(area, thick, active_start)

        # return the initial value of burn._oil_thickness - this is starting
        # thickness of the oil
        init_oil_th = self._weather_elements_helper(burn, avg_frac_water)

        # following should finally hold true for entire run
        assert np.allclose(amount, self.sc.weathering_data['burned'] +
                           self.sc['mass'].sum(), atol=1e-6)

        # want mass of oil thickness * area gives volume of oil-water so we
        # need to scale this by (1 - avg_frac_water)
        exp_burned = ((init_oil_th - burn._oil_thickness) * burn.area *
                      self.op.get_density())
        assert np.isclose(self.sc.weathering_data['burned'], exp_burned)

        mask = self.sc['status_codes'] == oil_status.burn

        # given LEs are discrete elements, we cannot add a fraction of an LE
        mass_per_le = self.sc['init_mass'][mask][0]
        exp_init_oil_mass = (burn.area * burn.thickness * (1 - avg_frac_water)
                             * self.op.get_density())
        assert (self.sc['init_mass'][mask].sum() - exp_init_oil_mass
                < mass_per_le and
                self.sc['init_mass'][mask].sum() - exp_init_oil_mass >= 0.0)

        exp_mass_remain = (burn._oil_thickness * burn.area *
                           self.op.get_density())
        mass_remain_for_burn_LEs = self.sc['mass'][mask].sum()
        assert np.allclose(exp_mass_remain, mass_remain_for_burn_LEs)

        print ('Current Thickness: {0:.3f}, '
               'Duration (hrs): {1:.3f}').format(burn._oil_thickness,
                                                 burn._burn_duration/3600)

    def test_elements_weather_faster_with_frac_water(self):
        '''
        Tests that avg_water_frac > 0 weathers faster. This is because the
        burn.thickness attribute now corresponds with thickness of boomed
        oil_water mixture. As water fraction increases, the thickness of the
        oil within the area that can burn is reduced by (1 - avg_frac_water),
        it therefore reaches the 2mm threshold fastest.

        TEST FAILS:
        todo: figure out if this is a good/valid test
        not sure about this since burn rate is slowed down by
        (1 - frac_water) so even though there less oil, the rate of burn is
        also reduced
        '''
        self.spill.set('num_elements', 500)
        area = (0.5 * self.volume)/1.
        burn1 = Burn(area, 1., active_start)
        burn2 = Burn(area, 1., active_start)
        burn3 = Burn(area, 1., active_start)

        init_th1 = self._weather_elements_helper(burn1)
        init_th2 = self._weather_elements_helper(burn2, avg_frac_water=0.3)
        init_th3 = self._weather_elements_helper(burn3, avg_frac_water=0.5)

        assert init_th1 > init_th2 > init_th3
        #assert (burn1._burn_duration > burn2._burn_duration >
        #        burn3._burn_duration)
