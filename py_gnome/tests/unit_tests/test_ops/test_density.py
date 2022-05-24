from gnome.ops.density import init_density, recalc_density
from gnome.ops.default_constants import default_water_density
from gnome.environment.water import Water
from gnome.spills.substance import NonWeatheringSubstance
from gnome.spills.gnome_oil import GnomeOil
from gnome.spill_container import SpillContainer
import numpy as np
from gnome.ops import weathering_array_types

from datetime import datetime, timedelta
from gnome import scripting as gs

def test_init_density():
    sc = SpillContainer()
    sc.substance = NonWeatheringSubstance()
    sc.prepare_for_model_run(weathering_array_types)
    sc._append_data_arrays(100)
    sc['density'] = np.ones(100)
    assert np.all(sc['density'] == 1)
    sc.substance.standard_density = 900
    default_water = Water()

    #when using a nonweathering substance, init_density should smiply init to
    # the substance standard_density.
    init_density(sc, 100, water=None)
    assert np.all(sc['density'] == 900)
    sc.substance.standard_density = 800
    init_density(sc, 100, water=default_water)
    assert np.all(sc['density'] == 800)

    sc.substance = GnomeOil('oil_ans_mp')
    
    init_density(sc, 100, water=None)
    rho1 = sc['density'].copy()
    default_water.temperature = 300
    init_density(sc, 100, water=default_water)
    assert np.all(sc['density'] < rho1)

def test_recalc_density():
# Setup with NonWeatheringSubstance, 100 LEs
    sc = SpillContainer()
#    nw_subs = NonWeatheringSubstance(standard_density=900)
#    sc.substance = nw_subs
# fix it **********
    spill = gs.surface_point_line_spill(num_elements=100,
                                        start_position=(0.0, 0.0, 0.0),
                                        release_time=datetime(2014, 1, 1, 0, 0),
                                        amount=100,
                                        units='bbl',
                                        substance = NonWeatheringSubstance(standard_density=900))  
    sc.spills.add(spill)
# fix it **********
    sc.prepare_for_model_run(weathering_array_types)
    sc._append_data_arrays(100)
    sc.mass_balance['avg_density'] = 0
    init_density(sc, 100, water=None, aggregate=False)
    sc['mass'][:] = 10 #necessary for avg_density

    assert np.all(sc['density'] == 900)
    default_water = Water()
    assert sc.mass_balance['avg_density'] == 0

    #Nonweathering density should not get recalculated.
    #Aggregation should still occur.
    recalc_density(sc, water=default_water, aggregate=True)
    assert np.all(sc['density'] == 900) 
    assert sc.mass_balance['avg_density'] == 900

    
#    new_subs = GnomeOil('oil_crude')
#    sc.rewind()
#    sc.substance = new_subs
# fix it **********
    sc = SpillContainer()
    spill = gs.surface_point_line_spill(num_elements=100,
                                        start_position=(0.0, 0.0, 0.0),
                                        release_time=datetime(2014, 1, 1, 0, 0),
                                        amount=100,
                                        units='bbl',
                                        substance=GnomeOil('oil_crude'))  
    sc.spills.add(spill)
# fix it **********
    sc.prepare_for_model_run(weathering_array_types)
    sc._append_data_arrays(100)
    sc.mass_balance['avg_density'] = 0
    sc['mass'][:] = 10 #necessary for avg_density and mass components
#    new_subs.initialize_LEs(100, sc, environment={'water':default_water})
    sc.substance.initialize_LEs(100, sc, environment={'water':default_water})

#    init_rho = new_subs.density_at_temp(default_water.get('temperature'))
    init_rho = sc.substance.density_at_temp(default_water.get('temperature'))
    assert np.all(sc['density'] == init_rho)
    new_water = Water(temperature=277)
    recalc_density(sc, water=new_water, aggregate=True)

    #temp went down so density goes up.
    assert np.all(sc['density'] > init_rho)
    assert sc.mass_balance['avg_density'] > init_rho

def test_sinker():
    sc = SpillContainer()
    new_subs = GnomeOil('oil_crude')
    new_subs.densities = [1004.0]
    new_subs.density_ref_temps = [288.15]
    sc.substance= new_subs
    sc.prepare_for_model_run(weathering_array_types)
    sc._append_data_arrays(100)

    w = Water()
    w.set('temperature', 288, 'K')
    w.set('salinity', 0, 'psu')
    init_density(sc, 100, water=w)
    assert np.all(sc['density'] == w.get('density'))



    
    
