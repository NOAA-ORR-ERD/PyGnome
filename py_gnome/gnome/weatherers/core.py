#!/usr/bin/env python

from datetime import timedelta

from gnome.movers.movers import Mover


class Weatherer(Mover):
    '''
       Base Weathering agent.  This is almost exactly like the base Mover
       in the way that it acts upon the model.  It contains the same API
       as the mover as well.
    '''
    def __init__(self, weathering, **kwargs):
        '''
           :param weathering: object that represents the weathering
                              properties of the substance that our
                              LEs are made up of.
        '''
        self.wc = weathering
        super(Weatherer, self).__init__(**kwargs)

    def __repr__(self):
        return ('Weatherer(active_start={0}, active_stop={1},'
                '\n    on={2}, active={3}'
                '\n    weathering={4}'
                '\n    )').format(self.active_start, self.active_stop,
                                  self.on, self.active, self.wc)

    def _get_active_time(self, time_step, model_time):
        'calculate the weathering time duration in seconds'
        model_end_time = model_time + timedelta(seconds=time_step)

        if self.active_stop < model_end_time:
            model_end_time = self.active_stop
        if self.active_start > model_time:
            model_time = self.active_start

        if model_end_time > model_time:
            return (model_end_time - model_time).total_seconds()
        else:
            return 0

    def _xform_inputs(self, sc, time_step, model_time):
        'make sure our inputs are a good fit for our calculations'
        if 'mass' not in sc:
            raise ValueError('No mass attribute available to calculate '
                             'weathering')

        time_step = self._get_active_time(time_step, model_time)
        return sc['mass'], time_step

    def get_move(self, sc, time_step, model_time):
        m0, time = self._xform_inputs(sc, time_step, model_time)
        return self.wc.weather(m0, time)
