#!/usr/bin/env python

"""
Example of a script using a "grid spill"

This was used to do a forecast for where to drop a whale.

We could then quickly see where a bunch of potential drop
locations would end up.

Also uses:

ROMS currents
point forecast winds
optionally -- gridded winds and a point forecast -- switching between them.

"""

import os
import gnome.scripting as gs

# define base directory
base_dir = os.path.dirname(__file__)
images_dir = os.path.join(base_dir, 'images')

WINDAGE = 0.03
print("running with windage:", WINDAGE)

# gs.set_verbose('info')
# pf = gs.PrintFinder()

rel_time = "2019-04-18T21:00:00"

model = gs.Model(name='Whale Drift',
                 time_step=3600,
                 start_time=rel_time,
                 duration=gs.days(3),
                 )

mapfile = gs.get_datafile(os.path.join(base_dir, 'coast.bna'))

model.map = gs.MapFromBNA(mapfile,
                          refloat_halflife=float("inf")
                          )

# nam = gs.WindMover("NAM_5K.nc")
# nam.active_range = (gs.MinusInfTime(),
#                     gs.asdatetime("2019-04-19T22:00"))
# model.movers += nam

# gfs = gs.WindMover("GFS_Global_0p5deg-4-18.nc")
# model.movers += gfs


windfile = gs.get_datafile(os.path.join(base_dir, '28NM SSW San Francisco CA-4-18.nws'))

forecast = gs.point_wind_mover_from_file(windfile)
#forecast.active_range = (gs.asdatetime("2019-04-19T22:01"),
#                          gs.InfTime())
model.movers += forecast



currentfile = gs.get_datafile(os.path.join(base_dir, 'CAROMS.nc'))

roms = gs.CurrentMover(currentfile)
# extrapolate the currents
roms.current.extrapolation_is_allowed = True
# roms.active_range = (gs.MinusInfTime(), gs.asdatetime("2019-04-19T20:00"))
model.movers += roms

model.spills += gs.grid_spill(bounds=((-122.3, 37.0),
                                      (-123.5, 38.0)),
                              resolution=60,
                              release_time=rel_time,
                              windage_range=(WINDAGE, WINDAGE),
                              windage_persist=(-1),
                              name='Whale Release Options',
                              )

model.movers += gs.RandomMover(diffusion_coef=10000)


model.outputters += gs.Renderer(map_filename=mapfile,
                                output_dir=images_dir,
                                image_size=(800, 800),
                                projection=None,
                                viewport=((-124.0, 36.0),
                                          (-122.0, 38.0)),
                                map_BB=None,
                                draw_back_to_fore=True,
                                draw_map_bounds=False,
                                draw_spillable_area=False,
                                formats=['gif'],
                                draw_ontop='forecast',
                                output_timestep=None,
                                output_zero_step=True,
                                output_last_step=True,
                                output_start_time=None,
                                on=True,
                                timestamp_attrib={},
                                )

outfilename = "whale_run_windage_{:.1f}.nc".format(WINDAGE * 100)
netcdf_file = os.path.join(base_dir, outfilename)
model.outputters += gs.NetCDFOutput(netcdf_file,
                                    which_data='standard',
                                    output_timestep=gs.hours(3),
                                    compress=True,
                                    surface_conc=None)

model.full_run()


