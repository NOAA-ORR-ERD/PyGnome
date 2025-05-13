"""
Example of surface currents from the US Navy GOFS: HYCOM

Rectangular grid

This is a small region around the mariana islands
"""

from pathlib import Path
import gnome.scripting as gs

HERE = Path(__file__).parent

# download the example file, if it's not already there
cur_file = gs.get_datafile(HERE / 'HYCOM.nc', subdir='')

curr = gs.GridCurrent.from_netCDF(HERE / cur_file)

# some grid info
min_lat = curr.grid.node_lat.min()
min_lon = curr.grid.node_lon.min()
max_lat = curr.grid.node_lat.max()
max_lon = curr.grid.node_lon.max()


bounds = ((min_lon, min_lat), (max_lon, max_lat))
print("bounds: ", bounds)

spill_location = (146.0, 15.0)

# make sure it runs in gnome
model = gs.Model(start_time=curr.data_start,
                 duration=curr.data_stop - curr.data_start,
                 )
model.movers += gs.CurrentMover(curr)
model.movers += gs.RandomMover()
model.spills += gs.point_line_spill(num_elements=100,
                                    start_position=spill_location,
                                    release_time=model.start_time
                                    )

model.outputters += gs.Renderer(
    output_timestep=gs.hours(1),
    output_dir=HERE,
    image_size=(800, 600),
    projection=None,
    viewport=bounds,
    formats=['gif'],
    )

model.full_run()


