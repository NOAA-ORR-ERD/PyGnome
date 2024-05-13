#!/usr/bin/env python

"""
This is a performance testing script for working with poorly chunked files

I"ts the result of a user noticing a huge performance degradation in WebGNOME

Which we confirmed and then Jay profiled and found it was file reasing access in the

Regular grid code -- which is ironic, as that *should* be the simplest

More detail in this issue:

https://gitlab.orr.noaa.gov/gnome/pygnome/-/issues/349

This is a script to test, using pretty much as simple as possible or a model
setup

It runs the model with the original chunked data file
then runs it again with the file converted to netcdf3,
which removes the chunking.

"""

import datetime
from pathlib import Path
import gnome.scripting as gs

chunked_file = "S7.nc"
nc3_file = "S7_nc3.nc"

if not Path(nc3_file).exists():
    print("nc3 file doesn't exist -- creating it")
    import subprocess
    subprocess.run(['nccopy', '-3', 'S7.nc', 'S7_nc3.nc'])


# orig spill location: (49.65715, 28.63329)
point1 = (49, 28)
point2 = (50, 29)

# times_in_currents = 2023-09-14 to 2023-09-22 09

model = gs.Model(start_time="2023-09-15 00:00:00",
                 duration=gs.hours(12),
                 time_step=gs.minutes(6),
                 )

currents = gs.GridCurrent.from_netCDF("S7.nc")
model.movers += gs.CurrentMover(currents)

model.spills += gs.point_line_spill(num_elements=1000,
                                    start_position=point1,
                                    release_time=model.start_time,
                                    end_release_time=None,
                                    end_position=point2,
                                    )

print("Starting the chunked run")
start = datetime.datetime.now()
model.full_run()
print(f"chunked files took: {datetime.datetime.now() - start}")

model.movers.clear()
currents = gs.GridCurrent.from_netCDF("S7_nc3.nc")
model.movers += gs.CurrentMover(currents)

print("Starting the nc3 run")
start = datetime.datetime.now()
model.full_run()
print(f"netcdf3 files took: {datetime.datetime.now() - start}")




