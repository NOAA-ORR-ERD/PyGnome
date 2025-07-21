"""
Example of a file from FVCOM in which teh UGRID-complinant grid spec was added.

IMPortnatnly, this need to be able to be saved to a save file and reloaded.
 - that was a bug at some point

"""

from pathlib import Path
import gnome.scripting as gs

HERE = Path(__file__).parent

current_filename = 'SSCOFS.ugrid.nc'
map_filename = 'DeceptionPass.bna'

# download the example file, if it's not already there
cur_file = gs.get_datafile(HERE / current_filename, 'gridded_test_files')

curr = gs.GridCurrent.from_netCDF(HERE / cur_file)

spill_location = (-122.6149358, 48.41271330)

# make sure it runs in gnome
model = gs.Model(start_time=curr.data_start,
                 time_step=gs.minutes(10),
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
    output_timestep=gs.minutes(10),
    output_dir=HERE,
    image_size=(800, 600),
    projection=None,
    #viewport=bounds,
    formats=['gif'],
    )

# model.full_run()

# save it out:

model.save("ugrid_compliant.gnome")

# and reload it:
model2 = gs.Model.load_savefile("ugrid_compliant.gnome")

model2.full_run()
