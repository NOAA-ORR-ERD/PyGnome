from core import Weatherer, HalfLifeWeatherer
from evaporation import Evaporation
from emulsification import Emulsification
from natural_dispersion import NaturalDispersion
from intrinsic import WeatheringData
from cleanup import Skimmer, Burn, Dispersion

__all__ = [Weatherer,
           HalfLifeWeatherer,
           Evaporation,
           Emulsification,
           NaturalDispersion,
           Skimmer,
           Burn,
           Dispersion,
           WeatheringData]


def weatherer_sort(weatherer):
    '''
    Returns an int describing the sorting order of the weatherer

    Weatherers are sorted as follows:

    0.  cleanup options including Skimmer, Burn
    1.  chemical dispersion - currently there is only one Dispersion
    2.  half-life - these are not used with following real weatherers but need
        to include them so sorting always works
    3.  evaporation - assign to all classes in this module
    4.  natural dispersion - does not yet exist
    5.  oil particle aggregation - what is this?
    6.  dissolution - does not exist
    7.  biodegradation - does not exist
    8.  emulsification
    '''
    if isinstance(weatherer, (Skimmer, Burn)):
        return 0

    if isinstance(weatherer, (Dispersion,)):
        return 1

    # NOTE:
    # For now there is only one Evaporation, Emulification etc, but could make
    # this more general if we have different models for Evaporation or
    # Emulsification etc
    #
    # Added HalfLifeWeatherer for completeness, not sure it will be used along
    # with other weatherers - it's more for simple model
    if isinstance(weatherer, (HalfLifeWeatherer,)):
        return 2

    if isinstance(weatherer, (Evaporation,)):
        return 3

    if isinstance(weatherer, (NaturalDispersion,)):
        return 4

    if isinstance(weatherer, (Emulsification,)):
        return 8
