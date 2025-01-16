"""
Example of using gridded data from

* meteorological model for winds
* an oceanographic model for currents

model results in netcdf files
"""

import gnome.scripting as gs
from pathlib import Path

data_dir = Path('example_files')

# setup the model
model = gs.Model(start_time="2023-03-03",
                 duration=gs.days(1),
                 time_step=gs.minutes(15)
                 )

# create and add map
map_fn = data_dir / 'mapfile.bna'
mymap = gs.MapFromBNA(map_fn, refloat_halflife=1)
model.map = mymap

# create and add a spill
spill = gs.point_line_spill(release_time="2023-03-03",
                            start_position=(-125, 48.0, 0),
                            num_elements=1000)
model.spills += spill

# create wind object and associated mover;
# add to model (also adds environment object)
fn = data_dir / 'gridded_wind.nc'
wind = gs.GridWind.from_netCDF(filename=fn)
wind_mover = gs.WindMover(wind)
model.movers += wind_mover

# # create a current mover (auto creates and adds environment object)
# fn = data_dir / 'gridded_current.nc'
# current_mover = gs.CurrentMover.from_netCDF(filename=fn)
# model.movers += current_mover

# create current object and associated mover;
# add to model (also adds environment object)
fn = data_dir / 'gridded_current.nc'
current = gs.GridCurrent.from_netCDF(filename=fn)
current_mover = gs.CurrentMover(current)
model.movers += current_mover

# Add random walk Diffusion
model.movers += gs.RandomMover(diffusion_coef=1e5)

# create a Renderer to see the output
renderer = gs.Renderer(mymap,
                       output_dir='./output',
                       output_timestep=gs.hours(6),
                       # set part of map to view
                       viewport=((-125.5, 47.5),
                                 (-124.0, 48.5)),
                       formats=['gif']
                       )

model.outputters += renderer

kmzout = gs.KMZOutput('output/gridded_example.kmz')
print("adding KMZ outputter")
model.outputters += kmzout

print("running the model: see output in the output dir")

model.full_run()

# Save it as a gnome save file:
model.save('gridded_example.gnome')



