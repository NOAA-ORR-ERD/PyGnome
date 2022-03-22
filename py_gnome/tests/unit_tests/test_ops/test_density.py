from gnome.ops.density import init_density, recalc_density
from gnome.ops.default_constants import default_water_density
from gnome.environment.water import Water
from gnome.spills.substance import NonWeatheringSubstance
from gnome.spills.gnome_oil import GnomeOil
from gnome.spill_container import SpillContainer
import numpy as np
from gnome.ops import weathering_array_types

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
    sc = SpillContainer()
    sc.substance = NonWeatheringSubstance()
    sc.prepare_for_model_run(weathering_array_types)
    sc._append_data_arrays(100)
    init_density(sc, 100, water=None)

    

    
    
