"""
Template for a script that can run a new example currents file.

It will set a  spill and run a trajectory, and output an animated GIF,
just so you can look at and makes sure it's doing the right thing.

You should be able to simply change the filenames at the top.

But you may need to also change the spill location,
to make sure it's a valid location.

Or any other customization that's particular to this example.

This template assumes a BNA file for a map:
you don't strictly need one, but it's nice for visualization.
"""

current_filename = 'FVCOM-Erie-OFS-subsetter.nc'
map_filename = 'lake_erie_islands_GOODS_shifted.bna'

# if you leave spill_location set to None, it will set it to
# the center of the grid, which may or may not be a valid location.
spill_location = None
# spill_location = (277.18, 41.67)


from pathlib import Path
import gnome.scripting as gs

HERE = Path(__file__).parent

# download the example file, if it's not already there
cur_file = gs.get_datafile(HERE / current_filename, 'gridded_test_files')

curr = gs.GridCurrent.from_netCDF(HERE / cur_file)

# some grid info
min_lat = curr.grid.node_lat.min()
min_lon = curr.grid.node_lon.min()
max_lat = curr.grid.node_lat.max()
max_lon = curr.grid.node_lon.max()


bounds = ((min_lon, min_lat), (max_lon, max_lat))
# print("bounds: ", bounds)

# Middle of the bounds -- valid??
center = ((max_lon + min_lon ) / 2, (max_lat + min_lat ) / 2)
# override if the mid_point isn't a valid
if spill_location is None:
    spill_location = center
print(f"{center=}")

# make sure it runs in gnome
model = gs.Model(start_time=curr.data_start,
                 # duration=gs.days(2),
                 duration=curr.data_stop - curr.data_start,
                 )
model.map = gs.MapFromBNA(HERE / map_filename)
model.movers += gs.CurrentMover(curr)
model.movers += gs.RandomMover()
model.spills += gs.point_line_spill(num_elements=100,
                                    start_position=spill_location,
                                    release_time=model.start_time
                                    )

model.outputters += gs.Renderer(
    map_filename=model.map,
    output_timestep=gs.hours(1),
    output_dir=HERE,
    image_size=(800, 600),
    projection=None,
    # viewport=bounds,
    formats=['gif'],
    )

model.full_run()
