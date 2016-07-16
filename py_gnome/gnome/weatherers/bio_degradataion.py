'''
model biodegradation process
'''
import numpy as np

from gnome.utilities.serializable import Serializable

from .core import WeathererSchema
from gnome.weatherers import Weatherer


class Biodegradation(Weatherer, Serializable):
    _state = copy.deepcopy(Weatherer._state)

    _schema = WeathererSchema

    def __init__(self, **kwargs):

        super(Biodegradation, self).__init__(**kwargs)


    def prepare_for_model_run(self, sc):
        '''
            Add biodegradation key to mass_balance if it doesn't exist.
            - Assumes all spills have the same type of oil
            - let's only define this the first time
        '''
        if self.on:
            super(Biodegradation, self).prepare_for_model_run(sc)
            sc.mass_balance['bio_degradation'] = 0.0

    def prepare_for_model_step(self, sc, time_step, model_time):
        '''
            Set/update arrays used by dispersion module for this timestep
        '''
        super(Biodegradation, self).prepare_for_model_step(sc, time_step, model_time)

        if not self.active:
            return


    def bio_degradate_oil(self, **kwargs):
        '''

        '''
        # TODO


    def weather_elements(self, sc, time_step, model_time):
        '''
            weather elements over time_step
        '''
        if not self.active:
            return

        if sc.num_released == 0:
            return

        # TODO

        sc.update_from_fatedataview()


    def serialize(self, json_='webapi'):
        """
            'water'/'waves' property is saved as references in save file
        """
        toserial = self.to_serialize(json_)
        schema = self.__class__._schema()
        serial = schema.serialize(toserial)

        # TODO

        return serial

    @classmethod
    def deserialize(cls, json_):
        """
            Append correct schema for water / waves
        """
        if not cls.is_sparse(json_):
            schema = cls._schema()

        # TODO
