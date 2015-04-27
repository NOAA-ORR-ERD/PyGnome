#!/usr/bin/env python
import copy
from datetime import timedelta

import numpy
np = numpy
from colander import (SchemaNode, drop, Bool)

from gnome.persist.validators import convertible_to_seconds
from gnome.persist.base_schema import ObjType
from gnome.persist.extend_colander import LocalDateTime

from gnome.array_types import mass_components
from gnome.utilities.serializable import Serializable

from gnome.movers.movers import Process, ProcessSchema


class WeathererSchema(ObjType, ProcessSchema):
    '''
    used to serialize object so need ObjType schema and it only contains
    attributes defined in base class (ProcessSchema)
    '''
    name = 'Weatherer'
    description = 'weatherer schema base class'


class Weatherer(Process):
    '''
       Base Weathering agent.  This is almost exactly like the base Mover
       in the way that it acts upon the model.  It contains the same API
       as the mover as well. Not Serializable since it does is partial
       implementation
    '''
    _state = copy.deepcopy(Process._state)
    _schema = WeathererSchema  # nothing new added so use this schema

    def __init__(self, **kwargs):
        '''
           :param weathering: object that represents the weathering
                              properties of the substance that our
                              LEs are made up of.
        '''
        super(Weatherer, self).__init__(**kwargs)

        # arrays that all weatherers will update - use this to ask
        self.array_types.update({'mass_components', 'mass'})

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'active_start={0.active_start!r}, '
                'active_stop={0.active_stop!r}, '
                'on={0.on}, '
                'active={0.active}'
                ')'.format(self))

    def prepare_for_model_run(self, sc):
        """
        Override for weatherers so they can initialize correct 'mass_balance'
        key and set initial value to 0.0
        """
        pass

    def weather_elements(self, sc, time_step, model_time):
        '''
        run the equivalent of get_move for weathering processes. It weathers
        each component and returns the mass remaining at end of time_step. It
        returns the mass in units of 'kg'
        '''
        raise NotImplementedError("All weatherers need to implement this "
            "method. It returns mass remaining for each component at end of "
            "time_step in 'kg' (SI units)")

    def _halflife(self, M_0, factors, time):
        'Assumes our factors are half-life values'
        half = np.float64(0.5)
        total_mass = M_0 * (half ** (time / factors))

        return total_mass

    def _exp_decay(self, M_0, lambda_, time):
        '''
        Exponential decay: x(t) = exp(lambda_*time)
        The lambda_ should be 'negative' in order for function to decay
        '''
        mass_remain = M_0 * np.exp(lambda_ * time)
        return mass_remain


class HalfLifeWeatherer(Weatherer):
    '''
    Give half-life for all components and decay accordingly
    '''
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

        arrays = ['mass_components', 'mass']
        for substance, data in sc.itersubstancedata(arrays):
            hl = self._halflife(data['mass_components'],
                                self.half_lives, time_step)
            data['mass_components'][:] = hl
            data['mass'][:] = data['mass_components'].sum(1)

        sc.update_from_fatedataview()