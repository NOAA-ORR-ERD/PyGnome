import logging
import numpy as np
from gnome.ops import default_constants
from .aggregated_data import aggregate as agg_func
from gnome.array_types import gat

logger = logging.getLogger(__name__)

def init_density(sc, num_released, water=None, substance=None, aggregate=True):
    '''
    Initializes the density of the last num_rel elements of a spill container (sc),
    taking into account environmental conditions.

    :param sc: spill container
    :param num_rel: int
    :param water: Water object to use. If None, uses default values
    :param substance: gnome.spills.substance.Substance or subclass thereof
    :param aggregate: Flag for whether to trigger mass balance updates in spill container
    '''
    substance = sc.substance if substance is None else substance
    water_temp = water_density = None
    #if substance.is_weatherable:
    if water is None:
        water_temp = default_constants.default_water_temperature
        water_density = default_constants.default_water_density
    else:
        water_temp = water.get('temperature', 'K')
        water_density = water.get('density')

    # Only the new elements need to be initialized
    sl = slice(-num_released, None, 1)
    density = substance.density_at_temp(water_temp)

    if density > water_density:
        msg = ("{0} will sink at given water temperature: {1:.1f} {2}. "
                "Setting density to water density"
                .format(substance.name,
                        water_temp,
                        'K')
        )
        logger.error(msg)
        sc['density'][sl] = water_density
    else:
        sc['density'][sl] = density

    if aggregate:
        agg_func(sc, num_released)

def recalc_density(sc, water=None, aggregate=True):
    '''
    Recalculates the density of the elements in a spill container. This is necessary
    if 'mass_components' have changed.
    :param sc: spill container
    :param water: Water object to use. If None, uses default
    :param aggregate: Flag for whether to trigger mass balance updates in spill container
    '''

    substance = sc.substance
    water_temp = water_rho = None

    if substance.is_weatherable: 
        for substance, data in sc.itersubstancedata(sc.array_types):  
            if len(data['mass']) == 0:
                continue    
                
            if water is None:
                water_temp = default_constants.default_water_temperature
                water_rho = default_constants.default_water_density
            else:
                water_temp = water.get('temperature', 'K')
                water_rho = water.get('density')

            if not substance.is_weatherable or len(data['density']) == 0:
                #substance isn't weatherable or no elements are present
                if aggregate:
                    agg_func(data, 0)
                return
                
            k_rho = _get_k_rho_weathering_dens_update(substance, water_temp)
            # sub-select mass_components array by substance.num_components.
            # Currently, physics for modeling multiple spills with different
            # substances is not correctly done in the same model. However,
            # let's put some basic code in place so the data arrays can infact
            # contain two substances and the code does not raise exceptions.
            # mass_components are zero padded for substance which has fewer
            # psuedocomponents. Subselecting mass_components array by
            # [mask, :substance.num_components] ensures numpy operations work
            mass_frac = (data['mass_components'][:, :substance.num_components]/data['mass'].reshape(len(data['mass']), -1))

            # check if density becomes > water, set it equal to water in this
            # case - 'density' is for the oil-water emulsion
            oil_rho = k_rho*(substance.component_density * mass_frac).sum(1)

            # oil/water emulsion density
            new_rho = (data['frac_water'] * water_rho +
                        (1 - data['frac_water']) * oil_rho)

            if np.any(new_rho > water_rho):
                new_rho[new_rho > water_rho] = water_rho
                logger.info('During density update, density is larger '
                                    'than water density - set to water density')

            data['density'] = new_rho
            data['oil_density'] = oil_rho

    sc.update_from_fatedataview(fate_status='all')

    # also initialize/update aggregated sc
    if aggregate:
        agg_func(sc, 0)

def _get_k_rho_weathering_dens_update(substance, temp_in_k):
    '''
    use lru_cache on substance. substance is expected to be a GnomeOil,
    if this object stays the same, then return the cached value for k_rho
    This depends on initial mass fractions, initial density and fixed
    component densities
    '''
    # update density/viscosity/relative_buoyancy/area for previously
    # released elements
    rho0 = substance.density_at_temp(temp_in_k)

    # dimensionless constant
    k_rho = (rho0 /
                (substance.component_density * substance.mass_fraction).sum())

    return k_rho
