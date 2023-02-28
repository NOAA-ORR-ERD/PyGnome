from functools import lru_cache

import numpy as np

from gnome.basic_types import fate, oil_status
from gnome.array_types import gat
from gnome.ops.viscosity import init_viscosity
from .sample_oils import _sample_oils
from .substance import Substance, SubstanceSchema


from gnome.persist import (NumpyArraySchema,
                           Int,
                           String,
                           Float,
                           SchemaNode,
                           drop)

from gnome.environment.water import WaterSchema


class Density(object):

    def __init__(self, kg_m_3, ref_temp_k, weathering=0):
        self.kg_m_3 = kg_m_3
        self.ref_temp_k = ref_temp_k
        self.weathering = weathering

    def __repr__(self):
        return ('<Density({0.kg_m_3} kg/m^3 at {0.ref_temp_k}K), '
                'w={0.weathering}>'
                .format(self))

def density_at_temp(ref_density, ref_temp_k, temp_k, k_rho_t=0.0008):
    '''
        Source: Adios2

        If we have an oil density at a reference temperature,
        then we can estimate what its density might be at
        another temperature.

        NOTE: need a reference for the coefficient of expansion
    '''
    return ref_density / (1.0 - k_rho_t * (ref_temp_k - temp_k))


def vol_expansion_coeff(rho_0, t_0, rho_1, t_1):
    '''
        Calculate the volumetric expansion coefficient of a liquid
        based on a set of two densities and their associated temperatures.
    '''
    if t_0 == t_1:
        k_rho_t = 0.0
    else:
        k_rho_t = (rho_0 - rho_1) / (rho_0 * (t_1 - t_0))

    return k_rho_t


def kvis_at_temp(ref_kvis, ref_temp_k, temp_k, k_v2=2100):
    '''
        Source: Adios2

        If we have an oil kinematic viscosity at a reference temperature,
        then we can estimate what its viscosity might be at
        another temperature.

        .. note::
              An analysis of the
              multi-KVis oils in our oil library suggest that a value of
              2100 would be a good default value for k_v2.
    '''
    return ref_kvis * np.exp((k_v2 / temp_k) - (k_v2 / ref_temp_k))


class GnomeOilSchema(SubstanceSchema):
    """
    Schema for Gnome Oil
    """
    standard_density = SchemaNode(Float(), read_only=True)
    water = WaterSchema(save_reference=True, test_equal=False)

    filename = SchemaNode(
        String(), missing=drop, isdatafile=False, save=False, test_equal=False)
    oil_name = SchemaNode(
        String(), missing=drop, save=True, update=True
    )
    '''schema for Oil object'''
    api = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    adios_oil_id = SchemaNode(
        String(), missing=drop, save=True, update=True
    )
    pour_point = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    solubility = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    bullwinkle_fraction = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    original_bullwinkle_fraction = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    bullwinkle_time = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    original_bullwinkle_time = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    emulsion_water_fraction_max = SchemaNode(
        Float(), missing=drop, save=True, update=True
    )
    densities = NumpyArraySchema(missing=drop, save=True, update=True)
    density_ref_temps = NumpyArraySchema(missing=drop, save=True, update=True)
    density_weathering = NumpyArraySchema(missing=drop, save=True, update=True)
    kvis = NumpyArraySchema(missing=drop, save=True, update=True)
    kvis_ref_temps = NumpyArraySchema(missing=drop, save=True, update=True)
    kvis_weathering = NumpyArraySchema(missing=drop, save=True, update=True)
    mass_fraction = NumpyArraySchema(missing=drop, save=True, update=True)
    boiling_point = NumpyArraySchema(missing=drop, save=True, update=True)
    molecular_weight = NumpyArraySchema(missing=drop, save=True, update=True)
    component_density = NumpyArraySchema(missing=drop, save=True, update=True)
#     sara_type = SequenceSchema(SchemaNode(String()),
#                                missing=drop,
#                                save=True,
#                                update=True)
    num_components = SchemaNode(Int(), missing=drop, save=True, update=True)


class GnomeOil(Substance):
    """
    Class to create an oil for use in Gnome
    """
    _schema = GnomeOilSchema
    _req_refs = ['water']

    def __init__(self,
                 oil_name=None,
                 filename=None,
                 water=None,
                 **kwargs):
        """
        Initialize a GnomeOil:

        :param oil_name=None: Name of one of the sample oils provided by:
                              ``gnome.spills.sample_oils``


        :param filename=None: filename (Path) of JSON file in the Adios Oil Database format.

        :param water=None: Water object with environmental conditions -- Deprecated.

        Additional keyword arguments will be passed to Substance: e.g.:
        ``windage_range``, ``windage_persist=None``,

        A GnomeOil can be initialized in three ways:

         1) From a sample oil name : ``GnomeOil(oil_name="sample_oil_name")`` the oils are available
            in gnome.spills.sample_oils

         2) From a JSON file in the ADIOS Oil Database format:
            ``GnomeOil(filename="adios_oil.json")`` usually records from the
            ADIOS Oil Database (https://adios.orr.noaa.gov)

         3) From the json : ``GnomeOil.new_from_dict(**json_)`` for loading
            save files, etc. (this is usually done under the hood)


        GnomeOil("sample_oil_name")        ---works for test oils from sample_oils only

        GnomeOil(oil_name="sample_oil_name")

        GnomeOil(filename="oil.json")      ---load from file using adios_db

        GnomeOil.new_from_dict(**json\_)    ---webgnomeclient, savefiles, etc.

        GnomeOil("invalid_name")           ---ValueError (not in sample oils)
        """
        try:
            super_kwargs = self._init_from_json(**kwargs)
        except TypeError:
            if oil_name is not None:
                if oil_name in _sample_oils:
                    # load from sample oil
                    oil_dict = _sample_oils[oil_name]
                    kwargs.update(oil_dict)
                else:
                    raise ValueError(f"{oil_name} not in sample_oils: options are:\n {_sample_oils.keys()} ")
            elif filename:
                self.from_adiosdb_file(filename, kwargs)

            super_kwargs = self._init_from_json(**kwargs)

        super(GnomeOil, self).__init__(**super_kwargs)

        self.filename = filename
        self.oil_name = oil_name
        self.water = water

        self._set_up_array_types()

    def from_adiosdb_file(self, filename, kwargs):
        try:
            import adios_db
        except ImportError as err:
            msg = "the adios_db package must be installed to use its json format"
            raise ImportError(msg) from err

        from adios_db.models.oil.oil import Oil as Oil_db
        from adios_db.computation.gnome_oil import make_gnome_oil

        oil_obj = Oil_db.from_file(filename)
        oil_name = oil_obj.metadata.name

        try:
            oil_dict = make_gnome_oil(oil_obj)
        except Exception as err:
            raise ValueError("selected oil is not suitable for Gnome") from err

        oil_dict['name'] = oil_name
        kwargs.update(oil_dict)


    def _set_up_array_types(self):
        # add the array types that this substance DIRECTLY initializes
        self.array_types.update({'density': gat('density'),
                                 'viscosity': gat('viscosity'),
                                 'mass_components': gat('mass_components')})
        self.array_types['mass_components'].shape = (self.num_components,)
        self.array_types['mass_components'].initial_value = (self.mass_fraction,)

    def _init_from_json(self,
                        # Physical properties
                        *,  # the rest are all keyword only
                        api,
                        pour_point,
                        solubility,  # kg/m^3
                        # emulsification properties
                        bullwinkle_fraction,
                        original_bullwinkle_fraction=None,
                        bullwinkle_time=None,
                        original_bullwinkle_time=None,
                        emulsion_water_fraction_max,
                        densities,
                        density_ref_temps,
                        density_weathering,
                        kvis,
                        kvis_ref_temps,
                        kvis_weathering,
                        # PCs:
                        mass_fraction,
                        boiling_point,
                        molecular_weight,
                        component_density,
                        sara_type=None,
                        adios_oil_id=None,
                        k0y=None,
                        num_components=None,
                        **kwargs):

        self.num_components = num_components if num_components else len(mass_fraction)

        self.api = api
        self.adios_oil_id = adios_oil_id
        self.densities = densities
        self.density_ref_temps = density_ref_temps
        self.density_weathering = density_weathering
        self.kvis = kvis
        self.kvis_ref_temps = kvis_ref_temps
        self.kvis_weathering = kvis_weathering

        self.bullwinkle_fraction = bullwinkle_fraction
        self.original_bullwinkle_fraction = (bullwinkle_fraction
                                             if original_bullwinkle_fraction is None
                                             else original_bullwinkle_fraction)
        self.bullwinkle_time = bullwinkle_time
        if self.bullwinkle_time is None:
            self.bullwinkle_time = -999.	# for the C code
        self.original_bullwinkle_time = (self.bullwinkle_time
                                         if original_bullwinkle_time is None
                                         else original_bullwinkle_time)

        self.emulsion_water_fraction_max = emulsion_water_fraction_max
        self.pour_point = pour_point
        self.solubility = solubility
        self._k_v2 = None

        # set the PC properties
        self._set_pc_values('mass_fraction', mass_fraction)
        self._set_pc_values('molecular_weight', molecular_weight)
        self._set_pc_values('boiling_point', boiling_point)
        self._set_pc_values('component_density', component_density)
#         if len(sara_type) == self.num_components:
#             self.sara_type = sara_type
#         else:
#             raise ValueError("You must have the same number of sara_type as PCs")
#
        self._k_v2 = None  # decay constant for viscosity curve
        self._visc_A = None  # constant for viscosity curve
        self.k0y = k0y

        return kwargs

    def __hash__(self):
        """
        needs to be hashable, so that it can be used in lru-cache

        Oils will only hash equal if they are the same object --
        that's limiting, but OK.
        """
        return id(self)

    def __deepcopy__(self, memo):
        """
        """
        return GnomeOil.deserialize(self.serialize())

    @classmethod
    def get_GnomeOil(self, oil_info, max_cuts=None):
        '''
        #fixme: what is oil_info ???

        Use this instead of get_oil_props
        '''
        return GnomeOil(oil_info)


    def to_dict(self, json_=None):
        json_ = super(GnomeOil, self).to_dict(json_=json_)

        return json_

    def initialize_LEs(self, to_rel, arrs, environment=None):
        '''
        :param to_rel - number of new LEs to initialize
        :param arrs - dict-like of data arrays representing LEs

        fixme:
               this shouldn't use water temp -- it should use
               standard density and STP temp -- and let
               weathering_data set it correctly

               .. note::
                   weathering data is currently broken
                   for initial setting
        '''
        water = self._pick_water(environment)
        init_viscosity(arrs, to_rel, water=water, aggregate=False)


        sl = slice(-to_rel, None, 1)

        fates = np.logical_and(arrs['positions'][sl, 2] == 0,
                               arrs['status_codes'][sl] == oil_status.in_water)
        if ('fate_status' in arrs):
            arrs['fate_status'][sl] = np.choose(fates, [fate.subsurf_weather, fate.surface_weather])

        # initialize mass_components
        arrs['mass_components'][sl] = (np.asarray(self.mass_fraction, dtype=np.float64) * (arrs['mass'][sl].reshape(len(arrs['mass'][sl]), -1)))

        super(GnomeOil, self).initialize_LEs(to_rel, arrs)

    def _set_pc_values(self, prop, values):
        """
        utility that sets a property to each pseudo component

        checks that it's the right size, and converts to an array
        """
        if len(values) != self.num_components:
            raise ValueError("must be the same number of {} as there "
                             "are pseudo components".format(prop))
        setattr(self, prop, np.array(values, dtype=np.float64))


    @lru_cache(2)
    def vapor_pressure(self, temp, atmos_pressure=101325.0):
        """
        the vapor pressure on the PCs at a given temperature
        water_temp and boiling point units are Kelvin

        :param temp: temperature in K

        :returns: vapor_pressure array in SI units (Pascals)

        ## Fixme: shouldn't this be in the Evaporation code?
        """
        D_Zb = 0.97
        R_cal = 1.987  # calories

        D_S = 8.75 + R_cal * np.log(self.boiling_point)
        C_2i = 0.19 * self.boiling_point - 18

        var = 1. / (self.boiling_point - C_2i) - 1. / (temp - C_2i)
        ln_Pi_Po = ((D_S * (self.boiling_point - C_2i) ** 2 /
                    (D_Zb * R_cal * self.boiling_point)) * var)
        Pi = np.exp(ln_Pi_Po) * atmos_pressure

        return Pi

    @classmethod
    def bounding_temperatures(cls, obj_list, temperature):
        '''
            General Utility Function

            From a list of objects containing a ref_temp_k attribute,
            return the object(s) that are closest to the specified
            temperature(s)

            Specifically:

                - We want the ones that immediately bound our temperature.

                - If our temperature is high and out of bounds of the temperatures
                  in our obj_list, then we return a range containing only the
                  highest temperature.

                - If our temperature is low and out of bounds of the temperatures
                  in our obj_list, then we return a range containing only the
                  lowest temperature.

            We accept only a scalar temperature or a sequence of temperatures
        '''
        temperature = np.array(temperature).reshape(-1, 1)

        if len(obj_list) <= 1:
            # range where the lowest and highest are basically the same.
            obj_list *= 2

        geq_temps = temperature >= [obj.ref_temp_k for obj in obj_list]

        high_and_oob = np.all(geq_temps, axis=1)
        low_and_oob = np.all(geq_temps ^ True, axis=1)

        rho_idxs0 = np.argmin(geq_temps, axis=1)
        rho_idxs0[rho_idxs0 > 0] -= 1
        rho_idxs0[high_and_oob] = len(obj_list) - 1

        rho_idxs1 = (rho_idxs0 + 1).clip(0, len(obj_list) - 1)
        rho_idxs1[low_and_oob] = 0

        return list(zip([obj_list[i] for i in rho_idxs0],
                   [obj_list[i] for i in rho_idxs1]))


    def get_densities(self):
        '''
            return a list of densities for the oil at a specified state
            of weathering.

            #fixme: this should not happen here!

            We include the API as a density if:

                - the specified weathering is 0

                - the culled list of densities does not contain a measurement
                  at 15C

        '''
        # fixme: why are we doing all this to make the same lists??
        densities = [d for d in self.densities]
        density_ref_temps = [d for d in self.density_ref_temps]
        weathering = [d for d in self.density_weathering]

        new_densities = []

        for x in range(0, len(densities)):
            if weathering[x] == 0: # also check None
                new_densities.append(Density(kg_m_3=densities[x],
                                             ref_temp_k=density_ref_temps[x],
                                             weathering=0.0))

        return sorted(new_densities, key=lambda d: d.ref_temp_k)

    def density_at_temp(self, temperature=288.15):
        ## fixme: this should use the Density object!
        '''
            Get the oil density at a temperature or temperatures.

            .. note:: This is all kruft left over from the estimating code.
                  At this point, a GnomeOil should already have what
                  it needs.

            .. note:: There is a catch-22 which prevents us from getting
                  the min_temp in some cases:

                      - To estimate pour point, we need viscosities

                      - If we need to convert dynamic viscosities to
                        kinematic, we need density at 15C

                      - To estimate density at temp, we need to estimate pour point

                      - ...and then we recurse
                      For this case we need to make an exception.
            .. note:: If we have a pour point that is higher than one or more
                  of our reference temperatures, then the lowest reference
                  temperature will become our minimum temperature.

            TODO:
                  We are getting rid of the argument that specifies a
                  weathering amount because it is currently implemented
                  in an unusably precise manner.  Robert would like us to
                  implement a means of interpolating density using a
                  combination of (temperature, weathering).  But the algorithm
                  for this is not defined at the moment.
        '''
        shape = None
        # fixme: define close to zero! don't count on isclose() default!
        densities = [d for d in self.get_densities()
                     if np.isclose(d.weathering, 0.0)]

        # set the minimum temperature to be the oil's pour point
        min_temp = np.min([d.ref_temp_k for d in densities] + [self.pour_point])

        if hasattr(temperature, '__iter__'):
            temperature = np.clip(temperature, min_temp, 1000.0)
            shape = temperature.shape
            temperature = temperature.reshape(-1)
        else:
            temperature = min_temp if temperature < min_temp else temperature

        ref_density, ref_temp_k = self._get_reference_densities(densities,
                                                                temperature)
        k_rho_t = self._vol_expansion_coeff(densities, temperature)

        rho_t = density_at_temp(ref_density, ref_temp_k,
                                temperature, k_rho_t)

        if len(rho_t) == 1:
            return rho_t[0]
        elif shape is not None:
            return rho_t.reshape(shape)
        else:
            return rho_t

    @property
    def standard_density(self):
        # fixme: this should simply be a set value
        #        computed on __init__
        '''
        Standard density is simply the density at 15C, which is the
        default temperature for density_at_temp()
        '''
        return float(self.density_at_temp(temperature=288.15))

    def _get_reference_densities(self, densities, temperature):
        '''
            Given a temperature, we return the best measured density,
            and its reference temperature, to be used in calculation.

            For our purposes, it is the density closest to the given
            temperature.
        '''
        closest_densities = self.bounding_temperatures(densities, temperature)

        try:
            # sequence of ranges
            density_values = np.array([[d.kg_m_3 for d in r]
                                       for r in closest_densities])
            ref_temp_values = np.array([[d.ref_temp_k for d in r]
                                        for r in closest_densities])

            greater_than = np.all((temperature > ref_temp_values.T).T, axis=1)

            density_values[greater_than, 0] = density_values[greater_than, 1]
            ref_temp_values[greater_than, 0] = ref_temp_values[greater_than, 1]

            return density_values[:, 0], ref_temp_values[:, 0]
        except TypeError:
            # single range
            density_values = np.array([d.kg_m_3 for d in closest_densities])
            ref_temp_values = np.array([d.ref_temp_k
                                        for d in closest_densities])

            if np.all(temperature > ref_temp_values):
                return density_values[1], ref_temp_values[1]
            else:
                return density_values[0], ref_temp_values[0]

    def _vol_expansion_coeff(self, densities, temperature):
        closest_densities = self.bounding_temperatures(densities, temperature)

        temperature = np.array(temperature)
        closest_values = np.array([[(d.kg_m_3, d.ref_temp_k)
                                    for d in r]
                                   for r in closest_densities])

        args_list = [[t for d in v for t in d]
                     for v in closest_values]
        k_rho_t = np.array([vol_expansion_coeff(*args)
                            for args in args_list])

        greater_than = np.all((temperature > closest_values[:, :, 1].T).T,
                              axis=1)
        less_than = np.all((temperature < closest_values[:, :, 1].T).T,
                           axis=1)

        if self.api > 30:
            k_rho_default = 0.0009
        else:
            k_rho_default = 0.0008

        k_rho_t[greater_than | less_than] = k_rho_default

        if k_rho_t.shape[0] == 1:
            return k_rho_t[0]
        else:
            return k_rho_t


    @classmethod
    def closest_to_temperature(cls, obj_list, temperature, num=1):
        '''
            General Utility Function

            From a list of objects containing a ref_temp_k attribute,
            return the object(s) that are closest to the specified
            temperature(s)

            We accept only a scalar temperature or a sequence of temperatures
        '''
        # fixme: this is NOT how to do this!
        if hasattr(temperature, '__iter__'):
            # we like to deal with numpy arrays as opposed to simple iterables
            temperature = np.array(temperature)

        # our requested number of objs can have a range [0 ... listsize-1]
        if num >= len(obj_list):
            num = len(obj_list) - 1

        temp_diffs = np.array([abs(obj.ref_temp_k - temperature)
                               for obj in obj_list]).T

        if len(obj_list) <= 1:
            return obj_list
        else:
            # we probably don't really need this for such a short list,
            # but we use a numpy 'introselect' partial sort method for speed
            try:
                # temp_diffs for sequence of temps
                closest_idx = np.argpartition(temp_diffs, num)[:, :num]
            except IndexError:
                # temp_diffs for single temp
                closest_idx = np.argpartition(temp_diffs, num)[:num]

            try:
                # sequence of temperatures result
                closest = [sorted([obj_list[i] for i in r],
                                  key=lambda x: x.ref_temp_k)
                           for r in closest_idx]
            except TypeError:
                # single temperature result
                closest = sorted([obj_list[i] for i in closest_idx],
                                 key=lambda x: x.ref_temp_k)

            return closest

    def kvis_at_temp(self, temp_k=288.15, weathering=0.0):
        """
        Compute the kinematic viscosity of the oil as a function of temperature

        :param temp_k: temperatures to compute at: can be scalar or array of values.
                       should be in Kelvin

        :param weathering: fraction weathered -- currently not implemented

        viscosity as a function of temp is given by:
        v = A exp(k_v2 / T)

        with constants determined from measured data
        """
        if weathering != 0.0:
            raise NotImplementedError("computing viscosity of weathered oil"
                                      "is not implemented yet")

        temp_k = np.asarray(temp_k)

        if self._k_v2 is None or self._visc_A is None:
            self.determine_visc_constants()

        return self._visc_A * np.exp(self._k_v2 / temp_k)

    def determine_visc_constants(self):
        '''
        viscosity as a function of temp is given by:

        v = A exp(k_v2 / T)

        The constants, A and k_v2 are determined from the viscosity data:

        If only one data point, a default value for k_vs is used:

           2100 K, based on analysis of data in the ADIOS database as of 2018

        If two data points, the two constants are directly computed

        If three or more, the constants are computed by a least squares fit.
        '''
        self._k_v2 = None # decay constant for viscosity curve
        self._visc_A = None

        kvis = [k for k, w in zip(self.kvis, self.kvis_weathering) if w == 0.0]
        kvis_ref_temps = [t for t, w in zip(self.kvis_ref_temps,
                                            self.kvis_weathering) if w == 0.0]

        if len(kvis) == 1:  # use default k_v2
            self._k_v2 = 2100.0
            self._visc_A = kvis[0] * np.exp(-self._k_v2 / kvis_ref_temps[0])
        else:
            # do a least squares fit to the data
            # viscs = np.array(kvis)
            # temps = np.array(kvis_ref_temps)
            b = np.log(kvis)
            A = np.c_[np.ones_like(b), 1.0 / np.array(kvis_ref_temps)]
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            self._k_v2 = x[1]
            self._visc_A = np.exp(x[0])
        return

    def get(self, prop):
        'get oil props'
        val = None
        try:
            val = getattr(self, prop)
        except AttributeError:
            try:
                val = getattr(self, prop)
            except Exception:
                pass

        return val
