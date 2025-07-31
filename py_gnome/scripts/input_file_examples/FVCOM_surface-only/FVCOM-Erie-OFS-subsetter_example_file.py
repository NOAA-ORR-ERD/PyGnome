"""
Example of a ROMS file with only surface currents.

This is from the NOAA COOPS WCOFS -- SF Bay entrance

Subsetted by the "OFS subsetter"

NOTE: the OFS FVCOM current files use 0-360 degree reference system.
      the map used has to also be in that reference system in order
      for it to work

"""

from pathlib import Path
import gnome.scripting as gs

HERE = Path(__file__).parent

current_filename = 'FVCOM-Erie-OFS-subsetter.nc'
map_filename = 'lake_erie_islands_GOODS_shifted.bna'

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
spill_location = center
# print(f"{center=}")
spill_location = (277.18, 41.67)

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
    #viewport=bounds,
    formats=['gif'],
    )

model.full_run()
