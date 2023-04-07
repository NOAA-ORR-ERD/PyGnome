"""
tests for the spatial release from polygons:

e.g. from the NESDIS MPSR reports
"""

import os
from pathlib import Path
import datetime
import numpy as np
import shapely
import pytest
import zipfile
import shapefile

from gnome.utilities.geometry import geo_routines
from gnome.spills.release import (PolygonRelease, NESDISRelease)

from gnome.environment import constant_wind, Water, Wind, Waves
from gnome.weatherers import FayGravityViscous, Evaporation, NaturalDispersion
from gnome import spill_container
from ..conftest import test_oil

from gnome import scripting as gs
from gnome.model import Model
from gnome.movers import SimpleMover
from gnome.outputters import WeatheringOutput

data_dir = Path(__file__).parent / "data_for_tests"

sample_nesdis_shapefile = data_dir / "NESDIS_files.zip"

class TestNESDISRelease:
    '''
    This class is to test compatibility between evaporation and spreading modules in PyGnome 
    '''
            
    @pytest.mark.parametrize('amount', [10, 1000.])    
    def test_model_run_integrated(self, amount):
    # step 0 ----- create a GNOME model object
        model = Model(time_step=datetime.timedelta(minutes=15),
                      duration=gs.days(1),
                      uncertain=True,
                      cache_enabled=False,
                      )  
    
    # step 1 ---- add environment objects
        model.environment += [constant_wind(1, 0), Water(temperature=25.0 + 273.15), Waves()]   
        
    # step 2 ---- add a spill object
        spill = self.polygon_spill(filename = sample_nesdis_shapefile,
                                   substance = gs.GnomeOil(test_oil),
                                   water = model.environment[-2],
                                   amount=amount,
                                   units='bbl')
        model.spills += spill        
        model.start_time = spill.release.release_time
    
    # step 3 ---- add a simple mover
        #model.movers += SimpleMover(velocity=(1., -1., 0.))
                
    # step 4 ---- add weatherer objects
        model.weatherers  += [Evaporation(model.environment[-2], model.environment[-3]), 
                              NaturalDispersion(model.environment[-1], model.environment[-2]),
                              FayGravityViscous()]    

    # step 5 ---- add an outputter
        model.outputters += WeatheringOutput()
        
    # step 6 ---- setup model run   
        model.full_run()

    def polygon_spill(self,
                      filename=None,
                      substance=None,
                      amount=0,
                      units='kg',
                      water=None,
                      on=True,
                      windage_range=None,
                      windage_persist=None,
                      name='Polygon Release'
                      ):
        '''
        Helper function returns a Spill object

        :param filename: NESDIS/Spatial oil data 
        
        :param amount=None: mass or volume of oil spilled
        :type amount: float

        :param units=None: units for amount spilled
        :type units: str

        :param water=None: water object

        :param tuple windage_range=(.01, .04): Percentage range for windage.
                                               Active only for surface particles
                                               when a mind mover is added
        :type windage_range: tuple

        :param windage_persist=900: Persistence for windage values in seconds.
                                        Use -1 for inifinite, otherwise it is
                                        randomly reset on this time scale
        :type windage_persist: int

        :param name='Surface Point/Line Spill': a name for the spill
        :type name: str
        '''
        
        release = NESDISRelease(filename)

        spill = self._setup_spill(release=release,
                             water=water,
                             substance=substance,
                             amount=amount,
                             units=units,
                             name=name,
                             on=on,
                             windage_range=windage_range,
                             windage_persist=windage_persist
                             )

        return spill

    def _setup_spill(self,
                     release,
                     water,
                     substance,
                     amount,
                     units,
                     name,
                     on,
                     windage_range,
                     windage_persist,
                    ):
        """
        set the windage on the substance, if it's provided

        otherwise simply passes everything in to the Spill
        """
        spill = gs.Spill(release=release,
                         water=water,
                         substance=substance,
                         amount=amount,
                         units=units,
                         name=name,
                         on=on)

        if substance is None:
            if windage_range is None:
                windage_range = (.01, .04)
            if windage_persist is None:
                windage_persist = 900
            spill.substance.windage_range = windage_range
            spill.substance.windage_persist = windage_persist
        elif windage_range is not None or windage_persist is not None:
            raise TypeError("You can not specify windage values if you specify a substance.\n"
                            "Set the windage on the substance instead")
        return spill    


class TestPointLineRelease:
    '''
    This class is to test compatibility between evaporation and spreading modules in PyGnome 
    '''
    
    @pytest.mark.parametrize('amount', [0, 1000.])    
    def test_model_run_integrated(self, amount):
    # step 0 ----- create a GNOME model object
        model = Model(time_step=datetime.timedelta(minutes=15),
                      duration=gs.days(1),
                      uncertain=True,
                      cache_enabled=False,
                      )  
    
    # step 1 ---- add environment objects
        model.environment += [constant_wind(1, 0), Water(temperature = 30.+273.15), Waves()]   
        
    # step 2 ---- add a spill object
        spill = gs.surface_point_line_spill(num_elements=1000,
                                            start_position=(0.0, 0.0, 0.0),
                                            release_time=datetime.datetime(2000, 1, 1, 1),
                                            amount=amount,
                                            substance=gs.GnomeOil(test_oil), 
                                            units='bbl')
        model.spills += spill        
        model.start_time = spill.release.release_time
    
    # step 3 ---- add a simple mover
        #model.movers += SimpleMover(velocity=(1., -1., 0.))
                
    # step 4 ---- add weatherer objects
        model.weatherers  += [Evaporation(model.environment[-2], model.environment[-3]), 
                              NaturalDispersion(model.environment[-1], model.environment[-2]),
                              FayGravityViscous()]    

    # step 5 ---- add an outputter
        model.outputters += WeatheringOutput()
        
    # step 6 ---- setup model run        
        model.full_run()
        