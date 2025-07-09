"""
Example of a ROMS 3D file with depth info that PYGNOME doesn't understand

But it works in surface only mode.
"""

from pathlib import Path
import gnome.scripting as gs

gs.PrintFinder()

HERE = Path(__file__).parent

# download the example file, if it's not already there
cur_file = gs.get_datafile(HERE / '3D_ROMS_example.nc', 'gridded_test_files')

curr = gs.GridCurrent.from_netCDF(HERE / cur_file)

# some grid info
min_lat = curr.grid.node_lat.min()
min_lon = curr.grid.node_lon.min()
max_lat = curr.grid.node_lat.max()
max_lon = curr.grid.node_lon.max()


bounds = ((min_lon, min_lat), (max_lon, max_lat))
# print("bounds: ", bounds)

spill_location = (-157.9, 21.2)

# make sure it runs in gnome
model = gs.Model(start_time=curr.data_start,
                 duration=gs.days(2),
                 # duration=curr.data_stop - curr.data_start,
                 )
model.map = gs.MapFromBNA(HERE / "oahu_coast.bna")
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
    #viewport=bounds,
    formats=['gif'],
    )

model.full_run()
