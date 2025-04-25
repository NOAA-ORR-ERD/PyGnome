"""
A script that demonstrates using uncertainly with currents.

This script requires the following data files, which can be found
in the example_scripts/example_files directory

"""


import gnome.scripting as gs
from pathlib import Path

base_dir = Path(__file__).parent
data_dir = base_dir / '../example_files'
#data_dir = Path('../example_files')

# setup the model
model = gs.Model(start_time="2023-03-03",
                 duration=gs.hours(24),
                 time_step=gs.minutes(15),
                 uncertain=True
                 )

# create and add map
map_filename = data_dir / 'mapfile.bna'
mymap = gs.MapFromBNA(map_filename)
model.map = mymap

# create and add a spill
spill = gs.point_line_spill(release_time="2023-03-03",
                                    start_position=(-125, 48.0, 0),
                                    num_elements=1000)
model.spills += spill

# create a current mover

cur_file = data_dir / 'gridded_current.nc'
# current_mover = gs.CurrentMover.from_netCDF(filename=cur_file,
#                                             uncertain_along=0.25,
#                                             uncertain_cross=0.1,
#                                             )

# cur = gs.SteadyUniformCurrent(speed=0.2,
#                               direction=135,
#                               )
cur = gs.GridCurrent.from_netCDF(filename=cur_file)
current_mover = gs.CurrentMover(current=cur,
                                uncertain_along=0.25,
                                uncertain_cross=0.1,
                                )
model.movers += current_mover

# Turn Diffusion down to see the effects of the uncertainty
model.movers += gs.RandomMover(diffusion_coef=1e1)

renderer = gs.Renderer(mymap,
                       output_dir='./output',
                       output_timestep=gs.hours(1),
                       # set part of map to view
                       viewport=((-125.10, 47.9),
                                 (-124.6, 48.10))
                       )

model.outputters += renderer

print("running the model: see output in the output dir")

model.full_run()

