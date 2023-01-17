import logging
import numpy as np
from gnome.ops import default_constants
from .aggregated_data import aggregate as agg_func

logger = logging.getLogger(__name__)


def init_viscosity(sc, num_released, water=None, aggregate=True):
    '''
    Initializes the viscosity of the last num_rel elements of a spill container (sc),
    taking into account environmental conditions.
    
    :param sc: spill container
    :param num_rel: int
    :param water: Water object to use. If None, uses default values
    :param aggregate: Flag for whether to trigger mass balance updates in spill container
    '''
    substance = sc.substance
    water_temp = water_density = None
    if substance.is_weatherable:
        if water is None:
            water_temp = default_constants.default_water_temperature
            water_density = default_constants.default_water_density
        else:
            water_temp = water.get('temperature', 'K')
            water_density = water.get('density')

        # Only the new elements need to be initialized
        sl = slice(-num_released, None, 1)
        sc['viscosity'][sl] = substance.kvis_at_temp(water_temp)
    if aggregate:
        agg_func(sc, num_released)

def recalc_viscosity(sc, water=None, aggregate=True):
    '''
    Recalculates the viscosity of the elements in a spill container.
    :param sc: spill container
    :param water: Water object to use. If None, uses default
    :param aggregate: Flag for whether to trigger mass balance updates in spill container
    '''

    substance = sc.substance
    water_temp = water_rho = None
    if substance.is_weatherable:
        if water is None:
            water_temp = default_constants.default_water_temperature
            water_rho = default_constants.default_water_density
        else:
            water_temp = water.get('temperature', 'K')
            water_rho = water.get('density')

        if not substance.is_weatherable or len(sc['density']) == 0:
            #substance isn't weatherable or no elements are present
            if aggregate:
                agg_func(sc, 0)
            return


        # following implementation results in an extra array called
        # fw_d_fref but is easy to read
        v0 = substance.kvis_at_temp(water_temp)

        if v0 is not None:
            kv1 = _get_kv1_weathering_visc_update(v0, default_constants.visc_curvfit_param)
            fw_d_fref = sc['frac_water'] / default_constants.visc_f_ref

            sc['viscosity'] = (v0 * np.exp(kv1 * sc['frac_evap']) * (1 + (fw_d_fref / (1.187 - fw_d_fref))) ** 2.49 )
            sc['oil_viscosity'] = (v0 * np.exp(kv1 * sc['frac_evap']))

    if aggregate:
        agg_func(sc, 0)

def _get_kv1_weathering_visc_update(v0, visc_curvfit_param):
    '''
    kv1 is constant.
    It defining the exponential change in viscosity as it weathers due to
    the fraction lost to evaporation/dissolution:
        v(t) = v' * exp(kv1 * f_lost_evap_diss)

    kv1 = sqrt(v0) * 1500
    if kv1 < 1, then return 1
    if kv1 > 10, then return 10

    Since this is fixed for an oil, it only needs to be computed once. Use
    lru_cache on this function to cache the result for a given initial
    viscosity: v0
    '''
    kv1 = np.sqrt(v0) * visc_curvfit_param

    if kv1 < 1:
        kv1 = 1
    elif kv1 > 10:
        kv1 = 10

    return kv1