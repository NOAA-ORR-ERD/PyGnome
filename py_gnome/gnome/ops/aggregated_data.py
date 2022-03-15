import logging
import numpy as np

from gnome.basic_types import oil_status, fate

logger = logging.getLogger(__name__)

def aggregate(sc, new_LEs=0):
    '''
    Updates the mass balance of the spill container, specifically the following:
        'avg_density',
        'avg_viscosity',
        'floating',
        'non_weathering',
        'amount_released',
    '''

    if sc['mass'].sum() > 0.0:
        sc.mass_balance['avg_density'] = \
            np.sum(sc['mass']/sc['mass'].sum() * sc['density'])
        sc.mass_balance['avg_viscosity'] = \
            np.sum(sc['mass']/sc['mass'].sum() * sc['viscosity'])
    else:
        logger.info("Sum of 'mass' array went to 0.0, cannot calculate avg density & viscosity")

    #in_water and 0.0 meters depth
    on_surface = ((sc['status_codes'] == oil_status.in_water) &
                    (sc['positions'][:,2] == 0.0))


    sc.mass_balance['floating'] = sc['mass'][on_surface].sum()
    sc.mass_balance['non_weathering'] = sc['mass'][sc['fate_status'] == fate.non_weather].sum()

    if new_LEs > 0:
        amount_released = np.sum(sc['mass'][-new_LEs:])

        if 'amount_released' in sc.mass_balance:
            sc.mass_balance['amount_released'] += amount_released
        else:
            sc.mass_balance['amount_released'] = amount_released