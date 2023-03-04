import gnome.scripting as gs
import os

#setup model
model = gs.Model(start_time="2023-03-03",
			 duration=gs.days(1),
			 time_step=gs.minutes(15)
			 )
			 
#create and add map
map_fn = os.path.join('example_files','mapfile.bna')
mymap = gs.MapFromBNA(map_fn, refloat_halflife=1) 
model.map = mymap

#create and add a spill
spill = gs.surface_point_line_spill(release_time="2023-03-03",
									start_position=(-125, 48.0, 0),
									num_elements=1000)
model.spills += spill
	
#create wind object and associated mover; add to model (also adds environment object	
fn = os.path.join('example_files','gridded_wind.nc')
wind = gs.GridWind.from_netCDF(filename=fn)
wind_mover = gs.WindMover(wind)
model.movers += wind_mover

#create current mover (auto creates and adds environment object)
fn = os.path.join('example_files','gridded_current.nc')
current_mover = gs.CurrentMover.from_netCDF(filename=fn)
model.movers += current_mover

renderer = gs.Renderer(map_filename=map_fn,
                       output_dir='./output',
					   output_timestep=gs.hours(6),
					   map_BB=mymap.get_map_bounding_box())

model.outputters += renderer

model.full_run()

