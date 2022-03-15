from gnome.ops.density import init_density, recalc_density
from gnome.ops.default_constants import default_water_density
from gnome.environment.water import Water
from gnome.spills.substance import NonWeatheringSubstance
from gnome.spills.gnome_oil import GnomeOil
import numpy as np

@pytest.mark.parametrize("aggregate", (False, True))
def test_init_density():
    #sc density is post-substance init (no environment effect) but pre-Model init
    sc = {'density': np.ones(100)}
    sc.substance = NonWeatheringSubstance()
    sc.substance.standard_density = 900
    sc.substance.initialize_LEs(100, sc)
    assert np.all(sc['density'] == 900)

    default_water = Water()

    #when using a nonweathering substance, this function does nothing.

    init_density(sc, 100, water=None)
    assert np.all(sc['density'] == 900)
    init_density(sc, 100, water=default_water)

    sc.substance = GnomeOil('oil_ans_mp')
    sc.substance.initialize_LEs()
    
    
