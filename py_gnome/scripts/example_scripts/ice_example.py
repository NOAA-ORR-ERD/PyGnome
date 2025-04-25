#!/usr/bin/env python

"""
A script that demonstrates using gnome in "ice infested waters"

The ice and current data are from a ROMS coupled ocean-ice model.
All files will be downloaded from https://gnome.orr.noaa.gov/py_gnome_testdata
if they are not available in the example_files directory
"""

import gnome.scripting as gs
from pathlib import Path

HERE = Path(__file__)

data_dir = Path('example_files')
output_dir = Path('output')

map_filename = data_dir / 'ak_arctic.bna'


# initialize the model
start_time = "1985-01-01T13:31"

model = gs.Model(start_time=start_time,
                 duration=gs.days(2),
                 time_step=gs.hours(1))

mapfile = gs.get_datafile(map_filename)

print('adding the map')
model.map = gs.MapFromBNA(mapfile, refloat_halflife=0.0)  # seconds

print('adding outputters')
renderer = gs.Renderer(mapfile, output_dir, image_size=(1024, 768))
renderer.set_viewport(((-165, 69), (-161.5, 70)))

model.outputters += renderer

netcdf_file = output_dir / 'ice_example.nc'
gs.remove_netcdf(netcdf_file)
model.outputters += gs.NetCDFOutput(netcdf_file,
                                    which_data='all')

print('adding a spill')
spill1 = gs.point_line_spill(num_elements=1000,
                             start_position=(-163.75,
                                             69.75,
                                             0.0),
                             release_time=start_time)

model.spills += spill1

print('getting the datafiles')
# option to use a file list
# fn = [gs.get_datafile(data_dir / 'arctic_avg2_0001_gnome.nc'),
#       gs.get_datafile(data_dir / 'arctic_avg2_0002_gnome.nc'),
#       ]

fn = gs.get_datafile(data_dir / 'arctic_avg2_0001_gnome.nc')

gt = {'node_lon': 'lon',
      'node_lat': 'lat'}

print('adding the ice current and ice wind movers')
ice_aware_curr = gs.IceAwareCurrent.from_netCDF(filename=fn,
                                                grid_topology=gt)
ice_aware_wind = gs.IceAwareWind.from_netCDF(filename=fn,
                                             grid=ice_aware_curr.grid,)
i_c_mover = gs.CurrentMover(current=ice_aware_curr)
i_w_mover = gs.WindMover(wind=ice_aware_wind)

# shifting to -180 to 180 longitude (to match the coordinate system of the map)
ice_aware_curr.grid.node_lon = gs.convert_longitude(ice_aware_curr.grid.node_lon[:], coord_system='-180--180')
model.movers += i_c_mover
model.movers += i_w_mover

print('adding an ice random mover')
model.movers += gs.IceAwareRandomMover(ice_concentration=ice_aware_curr.ice_concentration,
                                       diffusion_coef=50000)

# To include weathering use the ice_aware_wind for the Waves and Evaporation
# As long as you only use one wind it should work automatically

# to visualize the grid and currents
# renderer.add_grid(ice_aware_curr.grid)
# renderer.add_vec_prop(ice_aware_curr)

print("running the model:")

model.full_run()

print("Finished running the model: see output in the output dir")


# Save it as a gnome save file:
model.save('ice_example.gnome')

