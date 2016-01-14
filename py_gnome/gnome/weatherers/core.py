#!/usr/bin/env python
import copy

import numpy as np

from colander import SchemaNode

from gnome.persist.extend_colander import NumpyArray
from gnome.persist.base_schema import ObjType

from gnome.array_types import mass_components
from gnome.utilities.serializable import Serializable, Field
from gnome.exceptions import ReferencedObjectNotSet
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
        Base weatherer class; defines the API for all weatherers
        Passes optional arguments to base (Process) class via super. See base
        class for optional arguments:  `gnome.movers.mover.Process`

        adds 'mass_components', 'mass' to array_types since all weatherers
        need these.
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
        The lambda_ should be 'negative' in order for function to decay
        '''
        mass_remain = M_0 * np.exp(lambda_ * time)
        return mass_remain


class HalfLifeWeathererSchema(WeathererSchema):
    half_lives = SchemaNode(NumpyArray())


class HalfLifeWeatherer(Weatherer, Serializable):
    '''
    Give half-life for all components and decay accordingly
    '''
    _schema = HalfLifeWeathererSchema
    _state = copy.deepcopy(Weatherer._state)
    _state += Field('half_lives', save=True, update=True)

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
