'''
model evaporation process
'''






import copy

import numpy as np

from gnome import constants
from gnome.basic_types import oil_status
from gnome.array_types import gat
from gnome.exceptions import ReferencedObjectNotSet

from .core import WeathererSchema
from gnome.weatherers import Weatherer
from gnome.environment import (WindSchema,
                               WaterSchema)
from gnome.persist.base_schema import GeneralGnomeObjectSchema
from gnome.environment.gridded_objects_base import VectorVariableSchema


class EvaporationSchema(WeathererSchema):
    water = WaterSchema(
        save=True, update=True, save_reference=True
    )
    wind = GeneralGnomeObjectSchema(
        acceptable_schemas=[WindSchema, VectorVariableSchema],
        save=True, update=True, save_reference=True
    )

class Evaporation(Weatherer):
    _schema = EvaporationSchema
    _ref_as = 'evaporation'
    _req_refs = ['water', 'wind']

    def __init__(self,
                 water=None,
                 wind=None,
                 **kwargs):
        '''
        :param conditions: gnome.environment.Conditions object which contains
            things like water temperature
        :param wind: wind object for obtaining speed at specified time
        :type wind: Wind API, specifically must have get_value(time) method
        '''
        self.water = water
        self.wind = wind

        if water is not None and wind is not None:
            make_default_refs = False
        else:
            make_default_refs = True

        super(Evaporation, self).__init__(make_default_refs=make_default_refs, **kwargs)
        self.array_types.update({'positions': gat('positions'),
                                 'area': gat('area'),
                                 'evap_decay_constant': gat('evap_decay_constant'),
                                 'frac_water': gat('frac_water'),
                                 'frac_lost': gat('frac_lost'),
                                 'frac_evap': gat('frac_evap'),
                                 'init_mass': gat('init_mass')})

    def prepare_for_model_run(self, sc):
        '''
        add evaporated key to mass_balance
        for now also add 'density' key here
        Assumes all spills have the same type of oil
        '''
        # create 'evaporated' key if it doesn't exist
        # let's only define this the first time
        if self.on:
            super(Evaporation, self).prepare_for_model_run(sc)

            sc.mass_balance['evaporated'] = 0.0
            msg = ("{0._pid} init 'evaporated' key to 0.0").format(self)
            self.logger.debug(msg)

    def _mass_transport_coeff(self, points, model_time):
        '''
        Is wind a function of only model_time? How about time_step?
        at present yes since wind only contains timeseries data

            K = c * U ** 0.78 if U <= 10 m/s
            K = 0.06 * c * U ** 2 if U > 10 m/s

        If K is expressed in m/sec, then Buchanan and Hurford set c = 0.0025
        U is wind_speed 10m above the surface

        .. note:: wind speed is at least 1 m/s, unless there is an ice aware wind.

        .. note:: ice aware wind enforces minimum speed before applying coverage factor.
        '''
        wind_speed = self.get_wind_speed(points, model_time, min_val = 1, fill_value=1.0)
        #wind_speed[wind_speed < 1.0] = 1.0  # this is handled by get_wind_speed now
        c_evap = 0.0048     # if wind_speed in m/s
        return c_evap * wind_speed ** 0.78
#         return np.where(wind_speed <= 10.0,
#                         c_evap * wind_speed ** 0.78,
#                         0.06 * c_evap * wind_speed ** 2)

    def _set_evap_decay_constant(self, points, model_time, data, substance, time_step):
        # used to compute the evaporation decay constant
        K = self._mass_transport_coeff(points, model_time)
        water_temp = self.water.get('temperature', 'K')
        f_diff = 1.0
        if 'frac_water' in data:
            # frac_water content in emulsion will be a per element but is
            # currently not being set by anything. Fix once we initialize
            # and properly set frac_water
            f_diff = (1.0 - data['frac_water'])

        vp = substance.vapor_pressure(water_temp)
        #mw = substance.molecular_weight
        # evaporation expects mw in kg/mol, database is in g/mol
        mw = substance.molecular_weight / 1000.

        sum_mi_mw = (data['mass_components'][:, :len(vp)] / mw).sum(axis=1)

# term for Schmidt number
#        air_visc = 14.8e-6
#        D_water = 2.4e-5
#        (air_visc / D_water) ** (-2. / 3.) = 1.380277

        Schmidt = 1.380277 * (0.018 / mw) ** (1. / 3.)
# term for Schmidt number
        # d_numer = -1/rho * f_diff.reshape(-1, 1) * K * vp
        # d_denom = (data['thickness'] * constants.gas_constant *
        #            water_temp * sum_frac_mw).reshape(-1, 1)
        # data['evap_decay_constant'][:, :len(vp)] = d_numer/d_denom
        #
        # Do computation together so we don't need to make intermediate copies
        # of data - left sum_frac_mw, which is a copy but easier to
        # read/understand

        edc = ((-data['area'] * f_diff * K / (constants.gas_constant * water_temp * sum_mi_mw)).reshape(-1, 1)* vp * Schmidt)

        data['evap_decay_constant'][:, :len(vp)] = edc

        self.logger.debug(self._pid + 'max decay: {0}, min decay: {1}'.
                          format(np.max(data['evap_decay_constant']),
                                 np.min(data['evap_decay_constant'])))
        if np.any(data['evap_decay_constant'] > 0.0):
            raise ValueError("Error in Evaporation routine. One of the"
                             " exponential decay constant is positive")
        if np.any(np.isnan(data['evap_decay_constant'])):
            raise ValueError("Error in Evaporation routine. One of the"
                             " exponential decay constant is NaN")

    def weather_elements(self, sc, time_step, model_time):
        '''
        weather elements over time_step

        - sets 'evaporation' in sc.mass_balance
        - currently also sets 'density' in sc.mass_balance but may update
          this as we add more weatherers and perhaps density gets set elsewhere

        Following diff eq models rate of change each pseudocomponent of oil::

            dm(t)/dt = -(1 - fw) * A/B * m(t)

        Over a time-step, A, B, C are assumed constant. m(t) is the component
        mass at beginning of timestep; m(t + Dt) is mass at end of timestep::

            m(t + Dt) = m(t) * exp(-L * Dt)
            L := (1 - fw) * A/B

        Define properties for each pseudocomponent of oil and constants::

            vp: vapor pressure
            mw: molecular weight

        The following quantities are defined for a given blob of oil. The
        thickness of the blob is same for all LEs regardless of how many LEs
        are used to model the blob::

            area: area computed from fay spreading
            m_i: mass of component 'i'
            sum_m_mw: sum(m_i/mw_i) over all components

        effect of wind - mass transport coefficient::

            K: See _mass_transport_coeff()

        Finally, Evaporation of component 'i' for blob of oil::

            A = area * K * vp
            B = gas_constant * water_temp * sum_m_mw

        L becomes::
            L = (1 - fw) * area * K * vp/(gas_constant * water_temp * sum_m_mw)
        '''

        if not self.active or sc.num_released == 0 or not sc.substance.is_weatherable:
            return

        for substance, data in sc.itersubstancedata(self.array_types):
            if len(data['mass']) == 0:
                continue

            points = data['positions']
            # set evap_decay_constant array
            self._set_evap_decay_constant(points, model_time, data,
                                          substance, time_step)
            mass_remain = self._exp_decay(data['mass_components'], data['evap_decay_constant'], time_step)

            sc.mass_balance['evaporated'] += \
                np.sum(data['mass_components'][:, :] - mass_remain[:, :])

            # log amount evaporated at each step
            self.logger.debug(self._pid + 'amount evaporated for {0}: {1}'.
                              format(substance.name,
                                     np.sum(data['mass_components'][:, :] -
                                            mass_remain[:, :])))

            data['frac_evap'][:] += ((data['mass_components'][:, :] - mass_remain[:, :]).sum(1)/data['init_mass'])
            data['mass_components'][:] = mass_remain
            data['mass'][:] = data['mass_components'].sum(1)

            # add frac_lost
            data['frac_lost'][:] = 1 - data['mass']/data['init_mass']
        sc.update_from_fatedataview()


class BlobEvaporation(Evaporation):
    '''
    playing around with blob evaporation and time varying fay_area
    experimental code - not currently used by Model.
    Testing out the algorithm in ipython notebooks.

    See documentation in source code:
        gnome/documentation/evaporation/blob_evap.ipynb
    '''
    def _set_evap_decay_constant(self, model_time, data, substance, time_step):
        '''
        testing - for now assume only one spill and instantaneous spill
        '''
        # data should contain 'spill_num' as well so compute decay rate for
        # blobs released together
        # used to compute the evaporation decay constant
        K = self._mass_transport_coeff(model_time)
        water_temp = self.water.get('temperature', 'K')

        f_diff = 1.0
        if 'frac_water' in data:
            # frac_water content in emulsion will be a per element but is
            # currently not being set by anything. Fix once we initialize
            # and properly set frac_water
            f_diff = (1.0 - np.mean(data['frac_water']))

        vp = substance.vapor_pressure(water_temp)

        #mw = substance.molecular_weight
        # evaporation expects mw in kg/mol, database is in g/mol
        mw = substance.molecular_weight / 1000.


        # for now, for testing, assume instantaneous spill so get the
        # mass of the blob
        sum_mi_mw = (data['mass_components'][:, :].sum(0) / mw).sum()
        const = -(f_diff * K * vp /
                  (constants.gas_constant * water_temp * sum_mi_mw))

        # do it the same way for initial and all subsequent times
        blob_area = data['area'][0] * len(data['area'])
        t = data['age'][0]
        if t == 0:
            int_area = blob_area
        else:
            int_area = (blob_area * 2./3 *
                        ((t + time_step)**(3./2)/np.sqrt(t) - t))
        data['evap_decay_constant'][:, :] = const * int_area

        self.logger.debug(self._pid + 'max decay: {0}, min decay: {1}'.
                          format(np.max(data['evap_decay_constant']),
                                 np.min(data['evap_decay_constant'])))
