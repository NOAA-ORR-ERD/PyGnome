'''
Test evaporation module
'''
from datetime import timedelta

import pytest
import numpy as np

from gnome.environment import constant_wind, WaterProperties
from gnome.weatherers import Evaporation
from gnome.spill.elements import floating_weathering
from gnome.array_types import (windages,
                               mass_components,
                               density,
                               thickness,
                               mol,
                               evap_decay_constant)

from conftest import sample_sc_release


water_props = WaterProperties()


@pytest.mark.parametrize(('oil', 'temp', 'num_elems'),
                         [('oil_conservative', 311.15, 3),
                           #('FUEL OIL NO.6', 311.15)
                         ])
def test_evaporation(oil, temp, num_elems):
    '''
    still working on tests ..
    '''
    et = floating_weathering(substance=oil)
    et.water = water_props
    arrays = {'windages': windages,
              'mass_components': mass_components,
              'density': density,
              'thickness': thickness,
              'mol': mol,
              'evap_decay_constant': evap_decay_constant}

    sc = sample_sc_release(num_elements=num_elems,
                           element_type=et,
                           arr_types=arrays)

    time_step = 15. * 60
    model_time = (sc.spills[0].get('release_time') +
                  timedelta(seconds=time_step))

    evap = Evaporation(water_props, wind=constant_wind(1., 0))
    evap.prepare_for_model_run()
    evap.prepare_for_model_step(sc, time_step, model_time)
    mass_remain = evap.weather_elements(sc, time_step, model_time)
    assert np.all(sc['evap_decay_constant'] <= 0.0)

    #==========================================================================
    # print "\nDensity [kg/m^3]: "
    # print sc.spills[0].get('substance').density
    # print "\nOriginal mass_components"
    # print sc['mass_components']
    # print "\nMass Remaining: "
    # print mass_remain
    # print "\nDecay Rate:"
    # print sc['evap_decay_constant']
    # return (sc, mass_remain, evap)
    #==========================================================================


#(sc, mass_remain, evap) = test_evaporation('oil_conservative', 311.15, 2)
#(sc, mass_remain, evap) = test_evaporation('oil_jetfuels', 311.15, 2)
