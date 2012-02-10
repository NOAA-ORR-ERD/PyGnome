# Package wide type definitions
import numpy
from gnome import c_gnome


world_point = numpy.dtype([('p_long', numpy.int), ('p_lat', numpy.int)], align=True)
world_point3d = numpy.dtype([('p', world_point), ('z', numpy.double)], align=True)
world_rect = numpy.dtype([('lo_long', numpy.long), ('lo_lat', numpy.long), \
                            ('hi_long', numpy.long), ('hi_lat', numpy.long)], align=True)
le_rec = numpy.dtype([('le_units', numpy.int), ('le_key', numpy.int), ('le_custom_data', numpy.int), \
            ('p', world_point), ('z', numpy.double), ('release_time', numpy.uint), \
            ('age_in_hrs_when_released', numpy.double), ('clock_ref', numpy.uint), \
            ('pollutant_type', numpy.short), ('mass', numpy.double), ('density', numpy.double), \
            ('windage', numpy.double), ('droplet_size', numpy.int), ('dispersion_status', numpy.short), \
            ('rise_velocity', numpy.double), ('status_code', numpy.short), ('last_water_pt', world_point), ('beach_time', numpy.uint)], align=True)


status_not_released = c_gnome.status_not_released
status_in_water = c_gnome.status_in_water
status_on_land = c_gnome.status_on_land
status_off_maps = c_gnome.status_off_maps
status_evaporated = c_gnome.status_evaporated

disp_status_dont_disperse = c_gnome.disp_status_dont_disperse
disp_status_disperse = c_gnome.disp_status_disperse
disp_status_have_dispersed = c_gnome.disp_status_have_dispersed
disp_status_disperse_nat = c_gnome.disp_status_disperse_nat
disp_status_have_dispersed_nat = c_gnome.disp_status_have_dispersed_nat
disp_status_evaporate = c_gnome.disp_status_evaporate
disp_status_have_evaporated = c_gnome.disp_status_have_evaporated
disp_status_remove = c_gnome.disp_status_remove
sisp_status_have_removed = c_gnome.disp_status_have_removed
