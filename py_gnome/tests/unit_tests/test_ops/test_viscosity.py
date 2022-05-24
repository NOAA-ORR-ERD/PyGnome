from gnome.ops.viscosity import init_viscosity, recalc_viscosity
from gnome.environment.water import Water
from gnome.spills.substance import NonWeatheringSubstance
from gnome.spills.gnome_oil import GnomeOil
from gnome.spill_container import SpillContainer
import numpy as np
from gnome.ops import weathering_array_types

from datetime import datetime, timedelta
from gnome import scripting as gs

def test_init_viscosity():
    sc = SpillContainer()
    sc.substance = NonWeatheringSubstance()
    sc.prepare_for_model_run(weathering_array_types)
    sc._append_data_arrays(100)
    sc['viscosity'] = np.ones(100)
    assert np.all(sc['viscosity'] == 1)
    default_water = Water()

    #when using a nonweathering substance, init_viscosity should do nothing
    init_viscosity(sc, 100, water=None)
    assert np.all(sc['viscosity'] == 1)

    new_subs = sc.substance = GnomeOil('oil_ans_mp')
    
    init_viscosity(sc, 100, water=None)
    kv1 = sc['viscosity'].copy()
    default_water.temperature = 300
    init_viscosity(sc, 100, water=default_water)
    assert np.all(sc['viscosity'] < kv1)

def test_recalc_viscosity():
    sc = SpillContainer()
#    sc.substance = NonWeatheringSubstance()
# fix it **********
    sc = SpillContainer()
    spill = gs.surface_point_line_spill(num_elements=100,
                                        start_position=(0.0, 0.0, 0.0),
                                        release_time=datetime(2014, 1, 1, 0, 0),
                                        amount=100,
                                        units='bbl',
                                        substance = NonWeatheringSubstance())  
    sc.spills.add(spill)
# fix it **********
    sc.prepare_for_model_run(weathering_array_types)
    sc._append_data_arrays(100)
    sc['viscosity'] = np.ones(100)
    sc['mass'][:] = 10 #necessary for avg_viscosity and mass components
    sc.mass_balance['avg_viscosity'] = 0
    assert np.all(np.isclose(sc['viscosity'], 1))
    default_water = Water()

    #Nonweathering viscosity should not get recalculated.
    #Aggregation should still occur.
    recalc_viscosity(sc, water=default_water, aggregate=True)
    assert np.isclose(sc.mass_balance['avg_viscosity'], 1)
    assert np.all(np.isclose(sc['viscosity'], 1)) 


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
    sc['mass'][:] = 10 #necessary for avg_viscosity and mass components
    sc.mass_balance['avg_viscosity'] = 0
#    new_subs.initialize_LEs(100, sc, environment={'water':default_water})
    sc.substance.initialize_LEs(100, sc, environment={'water':default_water})
    
#    init_kv = new_subs.kvis_at_temp(default_water.get('temperature'))
    init_kv = sc.substance.kvis_at_temp(default_water.get('temperature'))
    assert np.all(sc['viscosity'] == init_kv)
    new_water = Water(temperature=277)
    recalc_viscosity(sc, water=new_water, aggregate=True)

    #temp went down so density goes up.
    assert np.all(sc['viscosity'] > init_kv)
    assert sc.mass_balance['avg_viscosity'] > init_kv
    