from core import Weatherer, HalfLifeWeatherer
from cleanup import Skimmer, Burn, ChemicalDispersion
from manual_beaching import Beaching
from evaporation import Evaporation
from natural_dispersion import NaturalDispersion
from dissolution import Dissolution
from emulsification import Emulsification
from weathering_data import WeatheringData
from spreading import Langmuir, FayGravityViscous, ConstantArea

'''
    Weatherers are to be ordered as follows:

    0.  half-life - This weatherer is just used for testing.  It is not used
        with the following real weatherers but we need to include it so
        sorting always works
    1.  cleanup options including Skimmer, Burn, Beaching
    2.  chemical dispersion
    3.  evaporation - assign to all classes in this module
    4.  natural dispersion
    5.  oil particle aggregation - what is this?
    6.  dissolution
    7.  biodegradation
    8.  emulsification
    9.  WeatheringData - defines initialize_data() method to initialize all
        weathering data arrays. In weather_elements, this updates data arrays
        corresponding with wd properties (density, viscosity)
    10. FayGravityViscous - initializes the 'fay_area' and 'area' array. Also
        updates the 'area' and 'fay_area' array
    11. Langmuir - modifies the 'area' array with fractional coverage based on
        langmuir cells.
'''
__all__ = [
           Weatherer,
           HalfLifeWeatherer,
           Skimmer,
           Burn,
           Beaching,
           ChemicalDispersion,
           Evaporation,
           NaturalDispersion,
           # OilParticleAggregation,
           Dissolution,
           # Biodegradation,
           Emulsification,
           WeatheringData,
           FayGravityViscous,
           ConstantArea,
           Langmuir,
           ]

weatherers_idx = dict(zip(__all__, range(len(__all__))))


def weatherer_sort(weatherer):
    '''
        Returns an int describing the sorting order of the weatherer
        or None if an order is not defined for the weatherer
    '''
    return weatherers_idx.get(weatherer.__class__)
