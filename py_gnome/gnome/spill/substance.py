import copy

from colander import Float, SchemaNode
import numpy as np
import warnings
from gnome.basic_types import fate, oil_status

from gnome.persist.base_schema import ObjTypeSchema, ObjType
from gnome.gnomeobject import GnomeId
from oil_library.oil_props import OilProps
from oil_library.factory import get_oil
from gnome.environment.water import Water

class SubstanceSchema(ObjTypeSchema):
    standard_density = SchemaNode(Float(), read_only=True)

    def __init__(self, unknown='preserve', *args, **kwargs):
        super(SubstanceSchema, self).__init__(*args, **kwargs)
        self.typ = ObjType(unknown)


class GnomeOil(GnomeId, OilProps):
    _schema = SubstanceSchema

    @classmethod
    def get_GnomeOil(self, oil_info, max_cuts=None):
        oil_ = get_oil(oil_info, max_cuts)
        return GnomeOil(oil_)


    def serialize(self, options={}):
        json_ = super(GnomeOil, self).serialize(options=options)
        substance_json = self.tojson()

        #the old 'prune_substance' function from Spill
        del substance_json['imported_record_id']
        del substance_json['estimated_id']

        for attr in ('kvis', 'densities', 'cuts',
                     'molecular_weights',
                     'sara_densities', 'sara_fractions'):
            for item in substance_json[attr]:
                for sub_item in ('id', 'oil_id', 'imported_record_id'):
                    if sub_item in item:
                        del item[sub_item]

        json_.update(substance_json)
        return json_

    @classmethod
    def new_from_dict(cls, dict_):
        substance = cls.get_GnomeOil(dict_)
        return substance

    @property
    def is_weatherable(self):
        if not hasattr(self, '_is_weatherable'):
            self._is_weatherable = True
        return self._is_weatherable

    @is_weatherable.setter
    def is_weatherable(self, val):
        self._is_weatherable = True if val else False

    def initialize_LEs(self, num_released, data, water=None):
        sl = slice(-num_released, None, 1)

        if water is None:
            #LEs released at standard temperature and pressure
            self.logger.warning('No water provided for substance initialization, using default Water object')
            water = Water()

        water_temp = self.water.get('temperature', 'K')
        density = self.density_at_temp(water_temp)
        if density > water.get('density'):
            msg = ("{0} will sink at given water temperature: {1} {2}. "
                   "Set density to water density"
                   .format(self.name,
                           self.water.get('temperature',
                                          self.water.units['temperature']),
                           self.water.units['temperature']))
            self.logger.error(msg)

            data['density'][sl] = self.water.get('density')
        else:
            data['density'][sl] = density

        data['init_mass'][sl] = data['mass'][sl]

        substance_kvis = self.kvis_at_temp(water_temp)
        if substance_kvis is not None:
            'make sure we do not add NaN values'
            data['viscosity'][sl] = substance_kvis


class NonWeatheringSubstance(GnomeId):
    _schema = SubstanceSchema

    def __init__(self,
                 standard_density=1000.0,
                 pour_point=273.15):
        '''
        Non-weathering substance class for use with ElementType.
        - Right now, we consider our substance to have default properties
          similar to water, which we can of course change by passing something
          in.

        :param standard_density=1000.0: The density of the substance, assumed
                                        to be measured at 15 C.
        :type standard_density: Floating point decimal value

        :param pour_point=273.15: The pour_point of the substance, assumed
                                  to be measured in degrees Kelvin.
        :type pour_point: Floating point decimal value
        '''
        self.standard_density = standard_density
        self._pour_point = pour_point

    @property
    def is_weatherable(self):
        if not hasattr(self, '_is_weatherable'):
            self._is_weatherable = False
        return self._is_weatherable

    @is_weatherable.setter
    def is_weatherable(self, val):
        self.logger.warn('This substance {0} cannot be set to be weathering')

    def initialize_LEs(self, num_released, data, water=None):
        sl = slice(-num_released, None, 1)
        data['density'][sl] = self.standard_density

        data['init_mass'][sl] = data['mass'][sl]

    def pour_point(self):
        '''
            We need to match the interface of the OilProps object, so we
            define this as a read-only function
        '''
        return self._pour_point

    def density_at_temp(self, temp):
        '''
            For non-weathering substance, we just return the standard density.
        '''
        return self.standard_density
