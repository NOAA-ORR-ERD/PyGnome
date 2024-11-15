"""
A script that demonstrates creating a spill with a polygon release.

This script requires map, current, and spill data files,  
which can be found in the example_scripts/example_files directory

"""
import gnome.scripting as gs
from gnome.basic_types import datetime_value_2d

from pathlib import Path
import numpy as np

import geopandas as gpd
import datetime

# Define base directory
base_dir = Path(__file__).parent
data_dir = Path('example_files')
output_dir = Path('output')

# Setup the model
start_time = "2024-10-28 11:00"
model = gs.Model(start_time=start_time,
                 duration=gs.hours(48),
                 time_step=gs.minutes(60)
                 )

# Create and add map
mapfile = data_dir / 'wa_coast.bna'
gnome_map = gs.MapFromBNA(mapfile, refloat_halflife=6)  # hours
model.map = gnome_map

# Create wind object and associated mover; add to model
series = np.zeros((5, ), dtype=datetime_value_2d)
start_time = gs.asdatetime(start_time)
series[0] = (start_time, (10, 45))
series[1] = (start_time + gs.hours(18), (10, 90))
series[2] = (start_time + gs.hours(30), (10, 135))
series[3] = (start_time + gs.hours(42), (10, 180))
series[4] = (start_time + gs.hours(54), (10, 225))

wind = gs.Wind(timeseries=series, units='m/s')
model.movers += gs.PointWindMover(wind)

# create current object and associated mover; add to model
cur_file = data_dir / 'wa_RTOFS_global.nc'
current = gs.GridCurrent.from_netCDF(filename=cur_file)
cur_mover = gs.CurrentMover(current)
model.movers += cur_mover

# Need some Diffusion
random_mover = gs.RandomMover(diffusion_coef=100000.0, uncertain_factor=2.0)
model.movers += random_mover

# Create Polygon spill
shapefile_path = data_dir / 'spatial_example.zip'
images_dir = output_dir / 'images'

# use filename or get list of polygons from the shapefile
#gdf = gpd.read_file(shapefile_path)
#polygons = gdf.geometry.tolist()

release = gs.PolygonRelease(filename=shapefile_path,
                            release_time=start_time,
                            #polygons=polygons
                            )
spill = gs.Spill(release=release,amount=1000,units='bbl')
model.spills += spill

# Option to use gs.polygon_release_spill utility to make it easier.
# spill = gs.polygon_release_spill(filename=shapefile_path, 
#                                  release_time=start_time)
# model.spills += spill

# Add outputters
model.outputters += gs.ShapeOutput(output_dir / 'shape_polygon_spill')
model.outputters += gs.Renderer(mapfile, 
                       images_dir, 
                       image_size=(800, 600),
                       viewport=((-126., 47.),
                                 (-124., 49.))
)

model.full_run()

# Save it as a gnome save file:
#model.save('polygon_spill_example.gnome')
