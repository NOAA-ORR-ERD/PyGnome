#!/usr/bin/env python






import numpy as np

from colander import SchemaNode

from gnome.persist.extend_colander import NumpyArray
from gnome.persist.base_schema import ObjTypeSchema
from gnome.array_types import gat

from gnome.utilities.time_utils import date_to_sec, sec_to_datetime
from gnome.exceptions import ReferencedObjectNotSet
from gnome.movers.movers import Process, ProcessSchema


class WeathererSchema(ProcessSchema):
    pass


class Weatherer(Process):
    '''
    Base Weathering agent.  This is almost exactly like the base Mover
    in the way that it acts upon the model.  It contains the same API
    as the mover as well. Not Serializable since it does is partial
    implementation
    '''
    _schema = WeathererSchema  # nothing new added so use this schema

    def __init__(self, **kwargs):
        '''
        Base weatherer class; defines the API for all weatherers
        Passes optional arguments to base (Process) class via super. See base
        class for optional arguments:  `gnome.movers.mover.Process`

        adds 'mass_components', 'mass' to array_types since all weatherers
        need these.
        '''
        super(Weatherer, self).__init__(**kwargs)

        # arrays that all weatherers will update - use this to ask
        self.array_types.update({'mass_components': gat('mass_components'),
                                 'fate_status': gat('fate_status'),
                                 'mass': gat('mass'),
                                 'init_mass': gat('init_mass')})

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'active_range={0.active_range!r}, '
                'on={0.on}, '
                'active={0.active})'
                .format(self))

    def initialize_data(self, sc, num_released):
        '''
        Let weatherers have a way to customize the initialization of
        data arrays. Currently, only some weatherers use this to customize
        initialization of data arrays. If movers also move towards this
        implementation, then move to 'Process' base class.
        '''
        pass

    def prepare_for_model_run(self, sc):
        """
        Override for weatherers so they can initialize correct 'mass_balance'
        key and set initial value to 0.0
        """
        if self.on:
            # almost all weatherers require wind, water, waves so raise
            # exception here if none is found
            for attr in ('wind', 'water', 'waves'):
                if hasattr(self, attr) and getattr(self, attr) is None:
                    msg = (attr + " object not defined for " +
                           self.__class__.__name__)
                    raise ReferencedObjectNotSet(msg)

    def weather_elements(self, sc, time_step, model_time):
        '''
        Run the equivalent of get_move for weathering processes. It modifies
        the SpillContainer's data arrays; most weatherers update
        'mass_components' and 'mass'

        Some objects do not implement this since they update arrays like 'area'
        in model_step_is_done()
        '''
        pass

    def _halflife(self, M_0, factors, time):
        'Assumes our factors are half-life values'
        half = np.float64(0.5)
        total_mass = M_0 * (half ** (time / factors))

        return total_mass

    def _exp_decay(self, M_0, lambda_, time):
        '''
        Exponential decay: x(t) = exp(lambda_*time)
        The `lambda_` should be 'negative' in order for function to decay
        '''
        mass_remain = M_0 * np.exp(lambda_ * time)
        return mass_remain

    def get_wind_speed(self, points, model_time, min_val = 0,
                       coord_sys='r', fill_value=1.0):
        '''
            Wrapper for the weatherers so they can get wind speeds
        '''
        if hasattr(self.wind,'ice_concentration'):
            retval = self.wind.at(points, model_time, min_val, coord_sys=coord_sys).reshape(-1)
        else:
            retval = self.wind.at(points, model_time, coord_sys=coord_sys).reshape(-1)
            retval[retval < min_val] = min_val

        if isinstance(retval, np.ma.MaskedArray):
            return retval.filled(fill_value)
        else:
            return retval

    def check_time(self, wind, model_time):
        '''
            Should have an option to extrapolate but for now we do by default

            TODO, FIXME: This function does not appear to be used by anything.
                         Removing it does not break any of the unit tests.
                         If it is not used, it should probably go away.
        '''
        new_model_time = model_time

        if wind is not None:
            if model_time is not None:
                timeval = date_to_sec(model_time)
                start_time = wind.get_start_time()
                end_time = wind.get_end_time()

                if end_time == start_time:
                    return model_time

                if timeval < start_time:
                    new_model_time = sec_to_datetime(start_time)

                if timeval > end_time:
                    new_model_time = sec_to_datetime(end_time)
            else:
                return model_time

        return new_model_time


class HalfLifeWeathererSchema(WeathererSchema):
    half_lives = SchemaNode(
        NumpyArray(), save=True, update=True
    )


class HalfLifeWeatherer(Weatherer):
    '''
    Give half-life for all components and decay accordingly
    '''
    _schema = HalfLifeWeathererSchema

    def __init__(self, half_lives=(15.*60, ), **kwargs):
        '''
        The half_lives are a property of HalfLifeWeatherer. If the

          len(half_lives) != gnome.array_types.mass_components.shape[0]

        then, only keep the number of elements of half_lives that equal the
        length of half_lives and consequently the mass_components array.
        The default is 5, it is possible to change default but not easily done.
        HalfLifeWeatherer is currently more for testing, so will change this if
        it becomes more widely used and there is a need for user to change
        default number of mass components.

        half_lives could be constants or could be something more complex like
        a function of time (not implemented yet). Not storing 'half_lives' in
        data_arrays since they are neither time-varying nor varying per LE.
        '''
        super(HalfLifeWeatherer, self).__init__(**kwargs)
        self.half_lives = half_lives

    @property
    def half_lives(self):
        return self._half_lives

    @half_lives.setter
    def half_lives(self, half_lives):
        self._half_lives = np.asarray(half_lives, dtype=np.float64)

    def weather_elements(self, sc, time_step, model_time):
        '''
        weather elements over time_step
        '''
        if not self.active:
            return

        if sc.num_released == 0:
            return

        for _, data in sc.itersubstancedata(self.array_types):
            hl = self._halflife(data['mass_components'],
                                self.half_lives, time_step)
            data['mass_components'][:] = hl
            data['mass'][:] = data['mass_components'].sum(1)

        sc.update_from_fatedataview()
