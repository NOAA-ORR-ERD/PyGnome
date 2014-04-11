#!/usr/bin/env python
import copy
from datetime import timedelta

import numpy
np = numpy
from colander import (SchemaNode, drop, Bool)

from gnome.persist.validators import convertible_to_seconds
from gnome.persist.base_schema import ObjType
from gnome.persist.extend_colander import LocalDateTime

from gnome.array_types import mass_components, half_lives
from gnome.utilities.serializable import Serializable

from gnome.movers.movers import Mover


class WeathererSchema(ObjType):
    on = SchemaNode(Bool(), default=True, missing=True)
    active_start = SchemaNode(LocalDateTime(), missing=drop,
                              validator=convertible_to_seconds)
    active_stop = SchemaNode(LocalDateTime(), missing=drop,
                             validator=convertible_to_seconds)


class Weatherer(Mover, Serializable):
    '''
       Base Weathering agent.  This is almost exactly like the base Mover
       in the way that it acts upon the model.  It contains the same API
       as the mover as well.
    '''
    _state = copy.deepcopy(Mover._state)
    _schema = WeathererSchema

    def __init__(self, **kwargs):
        '''
           :param weathering: object that represents the weathering
                              properties of the substance that our
                              LEs are made up of.
        '''
        super(Weatherer, self).__init__(**kwargs)
        self.array_types.update({'mass_components': mass_components,
                                 'half_lives': half_lives})

    def __repr__(self):
        return ('{0.__class__.__module__}.{0.__class__.__name__}('
                'active_start={0.active_start!r}, '
                'active_stop={0.active_stop!r}, '
                'on={0.on}, '
                'active={0.active}'
                ')'.format(self))

    def weather_elements(self, sc, time_step, model_time):
        '''
           Here we run get_move, and then apply the results to the elements
           in our spill container.  It just seems more intuitive that the
           weatherer control what happens to the elements instead of the model,
           as happens with the movers.
        '''
        hl = self.get_move(sc, time_step, model_time)
        sc['mass_components'][:] = hl
        sc['mass'][:] = hl.sum(1)

    def get_move(self, sc, time_step, model_time):
        m0, f, time = self._xform_inputs(sc, time_step, model_time)
        return self._halflife(m0, f, time)

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
        'calculate the weathering time duration in seconds'
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
