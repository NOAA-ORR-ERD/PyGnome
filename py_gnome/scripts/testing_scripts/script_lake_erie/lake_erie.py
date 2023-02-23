import gnome.scripting as gs
import numpy as np
import datetime
#from gnome.weatherers import (Emulsification,
#                              Evaporation,
#                              NaturalDispersion,
#                              ChemicalDispersion,
#                              Burn,
#                              Skimmer)
from gnome.outputters import Renderer

CURRENT_FILE = "Lakeerie_current_wind_ice_0002.nc"
currents = gs.GridCurrent.from_netCDF(CURRENT_FILE)
winds = gs.GridWind.from_netCDF(filename=CURRENT_FILE)
iceconcentrations=gs.IceConcentration.from_netCDF(CURRENT_FILE)
#print(f"{currents.data_start=}")
#print(f"{currents.data_stop=}")
#print("currents info:")

a= input("with ice or not? 0 means no ice, 1 means with ice ")
ice_or_not=int(a)
time = datetime.datetime(2022,4,1)
points = np.array([(-80.8, 42.2, 0.0)])
velocities = currents.at(points,time)
winds_detect=winds.at(points,time)

print(ice_or_not)
print(time)
print("velocities:", velocities)
print("wind:", winds_detect)


START_TIME = time
#START_TIME = "2022-04-01T00:00"

model = gs.Model(time_step=gs.minutes(30),
                 start_time=START_TIME,
                 duration=gs.days(10),
                 )


model.map = gs.MapFromBNA("lakeerie.bna")

current_mover = gs.PyCurrentMover(CURRENT_FILE)

bounds = current_mover.get_bounds()

#print(f"{bounds=}")

#model.spills += gs.grid_spill(bounds=bounds,
#                              resolution=100,
#                              release_time=START_TIME,
#                              # windage_range = (0.3, 0.3)
#                              )


#oil_file = '../alaska-north-slope_AD00020.json'
spill = gs.surface_point_line_spill(num_elements=1000,  # no need for a lot of elements for a instantaneous release
                                        start_position=(-80.8, 42.2, 0.0),
                                        release_time=START_TIME,
                                        amount=1000,
                                      #  substance=gs.GnomeOil(filename=oil_file),
                                      #  windage_range=(0.03, 0.03),
                                      #  windage_persist=(-1),
                                        units='bbl')
model.spills += spill

wind_mover = gs.PyWindMover(filename=CURRENT_FILE)


# The Ice
if ice_or_not > 0:
  print('adding the ice movers')
  
  fn = CURRENT_FILE
  ice_aware_curr = gs.IceAwareCurrent.from_netCDF(filename=CURRENT_FILE)
  ice_aware_wind = gs.IceAwareWind.from_netCDF(filename=fn,grid=ice_aware_curr.grid)
  i_c_mover = gs.PyCurrentMover(current=ice_aware_curr)
  i_w_mover = gs.PyWindMover(wind=ice_aware_wind)
  model.movers += i_c_mover
  model.movers += i_w_mover
  
  print('adding an Ice RandomMover:')
  model.movers += gs.IceAwareRandomMover(ice_concentration=ice_aware_curr.ice_concentration, diffusion_coef=50000)
                                                                            
  iceconcentration = iceconcentrations.at(points,time)   
  print("IceConcentration:", iceconcentration)
                                           
else:
  print("no ice")
  # this seems to be broken with the 3D currents :-(
  model.movers += current_mover
  # The wind
  model.movers += wind_mover
  print('adding a RandomMover:')
  model.movers += gs.RandomMover(diffusion_coef=50000)

  
  
## Weathering
#water = gs.Water(temperature=273.15) # 0 celsius
##wind = gs.constant_wind(20., 117, 'knots')
#waves = gs.Waves(ice_aware_wind, water)
#
#model.weatherers += Evaporation(water,ice_aware_wind)
#model.weatherers += Emulsification(waves)
#model.weatherers += NaturalDispersion(waves, water)



# Output
model.outputters += gs.NetCDFOutput(filename='lake_erie_results.nc',
                                    output_timestep=gs.hours(3) )

model.outputters += gs.Renderer(
    map_filename="lakeerie.bna",
    output_dir='./images',
    image_size=(1280, 1024),
    output_timestep=gs.hours(3), 
    # viewport=bounds,
    # formats=['gif'],
    )


## to visualize the grid and currents
#renderer = gs.Renderer('lakeerie.bna', output_dir='./images2', image_size=(1280, 1024))
## renderer.set_viewport(((-81, 42), (-78, 41)))
#model.outputters += renderer
##renderer.add_grid(ice_aware_curr.grid)
##renderer.add_vec_prop(ice_aware_curr)  #do not work


model.outputters += gs.OilBudgetOutput(filename='GNOME_oil_result.csv')


model.full_run()


