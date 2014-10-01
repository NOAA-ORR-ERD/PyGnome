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
    pass


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
        self.array_types.update({'mass_components': mass_components})

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

    def _xform_inputs(self, sc, time_step, model_time):
        'make sure our inputs are a good fit for our calculations'
        if 'mass_components' not in sc:
            raise ValueError('No mass attribute available to calculate '
                             'weathering')

        if 'half_lives' not in sc:
            raise ValueError('No half-lives attribute available to calculate '
                             'weathering')

        time_step = self._get_active_time(time_step, model_time)
        return sc['mass_components'], sc['half_lives'], time_step

    def _get_active_time(self, time_step, model_time):
        '''
        calculate the weathering time duration in seconds
        todo: if we want to resize time_step according to active_stop time,
        then we should perhaps be doing this for all processes - double check?
        Or open a ticket since it can wait
        And this can probably happen in prepare_for_model_step to set a local
        variable?
        '''
        if hasattr(time_step, 'total_seconds'):
            time_step = time_step.total_seconds()
        model_end_time = model_time + timedelta(seconds=time_step)

        if self.active_stop < model_end_time:
            model_end_time = self.active_stop
        if self.active_start > model_time:
            model_time = self.active_start

        if model_end_time > model_time:
            return (model_end_time - model_time).total_seconds()
        else:
            return 0

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
        self._hl = half_lives    # half lives input by the user

        # half_lives for mass_components, can only set this in
        # prepare_for_model_step
        self.half_lives = None

    def prepare_for_model_step(self, sc, time_step, model_time):
        '''
        update half_lives based on number of mass_components
        '''
        super(HalfLifeWeatherer, self).prepare_for_model_step(sc,
                                                              time_step,
                                                              model_time)
        num_pc = sc['mass_components'].shape[1]
        hl = np.zeros(num_pc, dtype=np.float64)
        hl[:] = np.Inf

        if self.half_lives is None:
            if num_pc < len(self._hl):
                hl = np.asarray(self.half_lives[:num_pc])
            elif num_pc > len(self._hl):
                hl[:len(self._hl)] = self._hl
            else:
                hl[:] = self._hl
            self.half_lives = hl

    def weather_elements(self, sc, time_step, model_time):
        '''
        weather elements over time_step
        '''
        if not self.active:
            return sc['mass_components']

        hl = self._halflife(sc['mass_components'], self.half_lives, time_step)
        return hl
