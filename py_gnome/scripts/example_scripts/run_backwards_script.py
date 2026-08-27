"""
Example of running gnome "backwards"

There are some limitations, but gnome can be run backwards, i.e.
use a negative timestep.

Note that the transport (movers) can simply be reversed --
The resulting path of the elements should be similar.

Some processes, e.g. weathering, diffusion, can not be reversed.

Weathering is disabled for a reverse run.

Diffusion is enabled, and will work, but it will act that same
way as going forward -- the elements will not gather together
back to a point.

Beaching and refloating work the same way as forwards as well,
but are not really a reversible process.

The drivers in this case are the same as for the gridded data example:

* meteorological model for winds
* an oceanographic model for currents

Making the model run backwards is as simple as
setting the run_backwards flag to True

Then you need to pay attention to model and spill
start times -- the start time and duration is set,
so the "start" or the run needs to be near the end
of the your driving data.

"""

import gnome.scripting as gs
from pathlib import Path

data_dir = Path(__file__).parent / 'example_files'

# time of the "end" position
model_start_time = "2023-03-04T00:00"
# setup the model
# making it run backwards is as simple as
# setting the run_backwards flag to True
model = gs.Model(start_time=model_start_time,
                 duration=gs.days(1),
                 time_step=gs.minutes(15),
                 run_backwards=True
                 )

# create and add map
map_fn = data_dir / 'mapfile.bna'
mymap = gs.MapFromBNA(map_fn, refloat_halflife=1)
model.map = mymap

# create and add a spill
spill = gs.point_line_spill(release_time=model_start_time,
                            start_position=(-125, 48.0, 0),
                            num_elements=1000)
model.spills += spill

# create wind object and associated mover;
# add to model (also adds environment object)
fn = data_dir / 'gridded_wind.nc'
wind = gs.GridWind.from_netCDF(filename=fn)
wind_mover = gs.WindMover(wind)
model.movers += wind_mover

# create a current mover (auto creates and adds environment object)
fn = data_dir / 'gridded_current.nc'
current_mover = gs.CurrentMover.from_netCDF(filename=fn)
model.movers += current_mover


# Add random walk Diffusion
model.movers += gs.RandomMover(diffusion_coef=1e5)

# create a Renderer to see the output
# saved as "mapfile_anim.gif"
renderer = gs.Renderer(mymap,
                       output_dir='./output',
                       output_timestep=gs.hours(6),
                       # set part of map to view
                       viewport=((-125.5, 47.5),
                                 (-124.0, 48.5)),
                       formats=['gif']  # animated gif
                       )

model.outputters += renderer

kmzout = gs.KMZOutput(
    'output/gridded_example.kmz',
    surface_conc=None,  # surface concentration doesn't really make sense for a backward run
)
print("adding outputters")
model.outputters += kmzout

netcdfout = gs.NetCDFOutput(
    'output/backwards_example.nc',
    which_data='standard',
    surface_conc=None, # surface concentration doesn't really make sense for a backward run
)

model.outputters += netcdfout

print("running the model: see output in the output dir")

model.full_run()

print("model done running")

# Save it as a gnome save file:
# that should be loadable in WebGNOME :-)
model.save('run_backwards.gnome')



