# Package wide type definitions
import numpy
from gnome import wind_mover


""" Basic types. Should each have a corresponding, equivalent type in the Cython."""
world_point_type = numpy.float64
world_point = numpy.dtype([('p_long', world_point_type),
                           ('p_lat', world_point_type)],
                           align=True)
world_point_3d = numpy.dtype([('p', world_point), ('z', numpy.double)], align=True)
world_rect = numpy.dtype([('lo_long', numpy.long), ('lo_lat', numpy.long), \
                            ('hi_long', numpy.long), ('hi_lat', numpy.long)], align=True)
le_rec = numpy.dtype([('le_units', numpy.int), ('le_key', numpy.int), ('le_custom_data', numpy.int), \
            ('p', world_point), ('z', numpy.double), ('release_time', numpy.uint), \
            ('age_in_hrs_when_released', numpy.double), ('clock_ref', numpy.uint), \
            ('pollutant_type', numpy.short), ('mass', numpy.double), ('density', numpy.double), \
            ('windage', numpy.double), ('droplet_size', numpy.int), ('dispersion_status', numpy.short), \
            ('rise_velocity', numpy.double), ('status_code', numpy.short), ('last_water_pt', world_point), ('beach_time', numpy.uint)], align=True)

wind_uncertain_rec = numpy.dtype([('randCos', numpy.float32), ('randSin', numpy.float32),], align=True)
velocity_rec = numpy.dtype([('u', numpy.double), ('v', numpy.double),], align=True)
time_value_pair = numpy.dtype([('time', numpy.uint32), ('value', velocity_rec),], align=True)

## could we use bit flags for status???
## and/or simpler status -- i.e 0 is "regular old LE that's moving... (I guess that's the status_in_water code)
## why are these found in the wind_move module?
status_not_released = wind_mover.status_not_released
status_in_water = wind_mover.status_in_water
status_on_land = wind_mover.status_on_land
status_off_maps = wind_mover.status_off_maps
status_evaporated = wind_mover.status_evaporated


disp_status_dont_disperse = wind_mover.disp_status_dont_disperse
disp_status_disperse = wind_mover.disp_status_disperse
disp_status_have_dispersed = wind_mover.disp_status_have_dispersed
disp_status_disperse_nat = wind_mover.disp_status_disperse_nat
disp_status_have_dispersed_nat = wind_mover.disp_status_have_dispersed_nat
disp_status_evaporate = wind_mover.disp_status_evaporate
disp_status_have_evaporated = wind_mover.disp_status_have_evaporated
disp_status_remove = wind_mover.disp_status_remove
sisp_status_have_removed = wind_mover.disp_status_have_removed
