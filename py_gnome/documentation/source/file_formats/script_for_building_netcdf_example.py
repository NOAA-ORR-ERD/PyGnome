"""
quick script for outputting and example netcdf file

This can be re-run if the format or documentation needs change
"""

import gnome.scripting as gs

model = gs.Model(duration=gs.hours(4),
                 time_step=gs.hours(1),
                 uncertain=False,
                 )

model.movers += gs.SimpleMover(velocity=(1, 1.5, 0))
model.movers += gs.RandomMover()

model.spills += gs.surface_point_line_spill(num_elements=10,
                                            start_position=(0.0, 0.0, 0.0),
                                            release_time=model.start_time,
                                            end_release_time=model.start_time + gs.hours(2),
                                            substance=gs.NonWeatheringSubstance(),
                                            )

outputter = gs.NetCDFOutput(filename="sample_gnome_output.nc",
                            which_data='standard',
                            surface_conc=None,
                            )

# output data can be removed (or added) to the arrays_to_output set.
# outputter.arrays_to_output.discard('density')
# outputter.arrays_to_output.discard('viscosity')

model.outputters += outputter

model.full_run()



