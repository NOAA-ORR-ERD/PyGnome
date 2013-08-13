import json
import os
#import datetime
from datetime import datetime, timedelta
import gnome.basic_types
import numpy as np
import sys

from pyramid.paster import bootstrap
from webgnome import WebPointSourceRelease, WebWindMover, util
from webgnome.model_manager import WebRandomMover, WebWind, WebGridCurrentMover
from webgnome.schema import ModelSchema


def main():
    """
    Configure a model with the parameters of the Boston script.
    """
    env = bootstrap('../development.ini')
    settings = env['registry'].settings
    location_file_dir = settings.location_file_dir
    pearl_harbor_dir = os.path.join(location_file_dir, 'pearl_harbor')
    pearl_harbor_data = os.path.join(pearl_harbor_dir, 'data')
    location_file = os.path.join(pearl_harbor_dir, 'location.json')

    if os.path.exists(location_file):
        message = 'File already exists:\n %s\nRemove? (y/n) ' % location_file
        if raw_input(message).lower() == 'y':
            os.unlink(location_file)
        else:
            print 'Cancelled.'
            return

    model = settings.Model.create()

    # Map file path is relative to package root
    map_file = os.path.join('location_files', 'pearl_harbor', 'data',
                            'pearl_harbor.bna')
    model.add_bna_map(map_file, {
        'refloat_halflife': 1 * 3600,  # seconds
    })

    start_time = datetime(2013, 1, 1, 1, 0)
    model.time_step = 900
    model.start_time = start_time
    model.duration = timedelta(days=1)
    model.uncertain = False

    # adding a wind mover

    series = np.zeros((3,), dtype=gnome.basic_types.datetime_value_2d)
    series[0] = (start_time,                      ( 4,   180) )
    series[1] = (start_time+timedelta(hours=12),  ( 2,   270) )
    series[2] = (start_time+timedelta(hours=24),  ( 4,   180) )
    
    wind = WebWind(timeseries=series, units='knots')
    model.environment += wind
    w_mover = WebWindMover(wind)
    model.movers += w_mover

    # adding a random mover
    random_mover = WebRandomMover(diffusion_coef=10000)
    model.movers += random_mover

    # adding a grid current mover:

    curr_file=os.path.join( pearl_harbor_data, "ch3d2013.nc")
    topology_file=os.path.join( pearl_harbor_data, "PearlHarborTop.dat")
    model.movers += gnome.movers.GridCurrentMover(curr_file,topology_file)


    # adding a spill

    spill = WebPointSourceRelease(num_elements=1000,
                                   start_position=(
                                       -157.97064, 21.331524, 0.0),
                                   release_time=start_time)

    model.spills += spill

    serialized_model = ModelSchema().bind().serialize(model.to_dict())
    model_json = json.dumps(serialized_model, default=util.json_encoder,
                            indent=4)

    with open(location_file, 'wb') as f:
        f.write(model_json)

    env['closer']()

if __name__ == '__main__':
    main()
