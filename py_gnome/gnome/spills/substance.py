
from backports.functools_lru_cache import lru_cache

from colander import Int, Schema, String, Float, SchemaNode, SequenceSchema, Boolean, drop
import numpy as np
import os

from gnome.basic_types import fate, oil_status
from gnome.array_types import gat

from gnome.persist.base_schema import (ObjTypeSchema,
                                       ObjType,
                                       GeneralGnomeObjectSchema)

from gnome.persist.extend_colander import NumpyArraySchema
from gnome.gnomeobject import GnomeId
from gnome.environment.water import Water, WaterSchema

from .sample_oils import _sample_oils
from .initializers import (floating_initializers,
                                      InitWindagesSchema,
                                      DistributionBaseSchema)

class SubstanceSchema(ObjTypeSchema):
    initializers = SequenceSchema(
        GeneralGnomeObjectSchema(
            acceptable_schemas=[InitWindagesSchema,
                                DistributionBaseSchema
                                ]
        ),
        save=True, update=True, save_reference=True
    )
    is_weatherable = SchemaNode(Boolean(), read_only=True)


class NonWeatheringSubstanceSchema(SubstanceSchema):
    standard_density = SchemaNode(Float(), read_only=True)


class Substance(GnomeId):
    _schema = SubstanceSchema
    _ref_as = 'substance'

    def __init__(self,
                 initializers=None,
                 windage_range=(.01, .04),
                 windage_persist=900,
                 *args,
                 **kwargs):
        super(Substance, self).__init__(*args, **kwargs)
        if not initializers:
            initializers = floating_initializers(windage_range=windage_range,
                                                 windage_persist=windage_persist)
        self.initializers = initializers
        # add the types from initializers
        self._windage_init = None
        for i in self.initializers:
            self.array_types.update(i.array_types)
            if 'windages' in i.array_types:
                self._windage_init = i
        if windage_range != (.01, .04):
            self.windage_range = windage_range
        if windage_persist != 900:
            self.windage_persist = windage_persist

    @property
    def all_array_types(self):
        '''
        Need to add array types from Release and Substance
        '''
        arr = self.array_types.copy()
        for init in self.initializers:
            arr.update(init.all_array_types)
        return arr

    @property
    def is_weatherable(self):
        if not hasattr(self, '_is_weatherable'):
            self._is_weatherable = True
        return self._is_weatherable

    @is_weatherable.setter
    def is_weatherable(self, val):
        self._is_weatherable = True if val else False
    '''
    Windage range/persist are important enough to receive properties on the
    Substance.
    '''
    @property
    def windage_range(self):
        if self._windage_init:
            return self._windage_init.windage_range
        else:
            raise ValueError('No windage initializer on this substance')

    @windage_range.setter
    def windage_range(self, val):
        if self._windage_init:
            if np.any(np.asarray(val) < 0) or np.asarray(val).size != 2:
                raise ValueError("'windage_range' >= (0, 0). "
                                 "Nominal values vary between 1% to 4%. "
                                 "Default windage_range=(0.01, 0.04)")
            self._windage_init.windage_range = val
        else:
            raise ValueError('No windage initializer on this substance')

    @property
    def windage_persist(self):
        if self._windage_init:
            return self._windage_init.windage_persist
        else:
            raise ValueError('No windage initializer on this substance')

    @windage_persist.setter
    def windage_persist(self, val):
        if self._windage_init:
            if val == 0:
                raise ValueError("'windage_persist' cannot be 0. "
                                 "For infinite windage, windage_persist=-1 "
                                 "otherwise windage_persist > 0.")
            self._windage_init.windage_persist = val
        else:
            raise ValueError('No windage initializer on this substance')

    def get_initializer_by_name(self, name):
        ''' get first initializer in list whose name matches 'name' '''
        init = [i for i in enumerate(self.initializers) if i.name == name]

        if len(init) == 0:
            return None
        else:
            return init[0]

    def has_initializer(self, name):
        '''
        Returns True if an initializer is present in the list which sets the
        data_array corresponding with 'name', otherwise returns False
        '''
        for i in self.initializers:
            if name in i.array_types:
                return True

        return False

    def initialize_LEs(self, to_rel, arrs):
        '''
        :param to_rel - number of new LEs to initialize
        :param arrs - dict-like of data arrays representing LEs
        '''
        for init in self.initializers:
            init.initialize(to_rel, arrs, self)

    def _attach_default_refs(self, ref_dict):
        for i in self.initializers:
            i._attach_default_refs(ref_dict)
        return GnomeId._attach_default_refs(self, ref_dict)


class NonWeatheringSubstance(Substance):
    _schema = NonWeatheringSubstanceSchema

    """
    The simplest substance that can be used with the model

    It can not be weathereed, but does have basic properties for transport:

    Windage, density, etc.
    """

    def __init__(self,
                 standard_density=1000.0,
                 **kwargs):
        """
        Initialize a non-weathering substance.

        All parameters are optional

        :param standard_density=1000.0: The density of the substance, assumed
                                        to be measured at 15 C.
        :type standard_density: Floating point decimal value

        :param pour_point=273.15: The pour_point of the substance, assumed
                                  to be measured in degrees Kelvin.
        :type pour_point: Floating point decimal value
        """

        super(NonWeatheringSubstance, self).__init__(**kwargs)
        self.standard_density = standard_density
        self.array_types.update({
            'density': gat('density'),
            'fate_status': gat('fate_status')})

    @property
    def is_weatherable(self):
        if not hasattr(self, '_is_weatherable'):
            self._is_weatherable = False
        return self._is_weatherable

    @is_weatherable.setter
    def is_weatherable(self, val):
        self.logger.warning('This substance {0} cannot be set to be weathering')

    def initialize_LEs(self, to_rel, arrs):
        '''
        :param to_rel - number of new LEs to initialize
        :param arrs - dict-like of data arrays representing LEs
        '''
        sl = slice(-to_rel, None, 1)
        arrs['density'][sl] = self.standard_density
        if ('fate_status' in arrs):
            arrs['fate_status'][sl] = fate.non_weather
        super(NonWeatheringSubstance, self).initialize_LEs(to_rel, arrs)

    def density_at_temp(self, temp=273.15):
        '''
            For non-weathering substance, we just return the standard density.
        '''
        return self.standard_density

# so old save files will work
# this should be removed eventually ...
from .gnome_oil import GnomeOil


