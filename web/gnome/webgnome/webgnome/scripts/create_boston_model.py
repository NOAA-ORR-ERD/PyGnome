import json
import os
import datetime
import gnome.basic_types
from gnome.environment import Tide
import numpy
import sys

from pyramid.paster import bootstrap
from webgnome import WebSurfaceReleaseSpill, WebWindMover, util
from webgnome.model_manager import WebRandomMover, WebWind
from webgnome.schema import ModelSchema


def main():
    """
    Configure a model with the parameters of the Boston script.
    """
    env = bootstrap('../development.ini')
    settings = env['registry'].settings
    location_file_dir = settings.location_file_dir
    boston_dir = os.path.join(location_file_dir, 'boston')
    boston_data = os.path.join(boston_dir, 'data')
    location_file = os.path.join(boston_dir, 'location.json')

    if os.path.exists(location_file):
        message = 'File already exists:\n %s\nRemove? (y/n) ' % location_file
        if raw_input(message).lower() == 'y':
            os.unlink(location_file)
        else:
            print 'Cancelled.'
            return

    model = settings.Model.create()

    # Map file path is relative to package root
    map_file = os.path.join('location_files', 'boston', 'data',
                            'MassBayMap.bna')
    model.add_bna_map(map_file, {
        'refloat_halflife': 1 * 3600,  # seconds
    })

    start_time = datetime.datetime(2013, 3, 12, 10, 0)
    model.time_step = 900
    model.start_time = start_time
    model.duration = datetime.timedelta(days=1)
    model.uncertain = False

    model.movers += gnome.movers.RandomMover(diffusion_coef=100000)

    # adding a wind mover

    series = numpy.zeros((2,), dtype=gnome.basic_types.datetime_value_2d)
    series[0] = (start_time, (5, 180))
    series[1] = (start_time + datetime.timedelta(hours=18), (5, 180))

    wind = WebWind(timeseries=series, units='m/s')
    model.environment += wind
    w_mover = WebWindMover(wind)
    model.movers += w_mover

    # adding a random mover
    random_mover = WebRandomMover(diffusion_coef=100000)
    model.movers += random_mover

    # adding a cats shio mover:

    curr_file = os.path.join(boston_data, "EbbTides.CUR")
    shio_file = os.path.join(boston_data, "EbbTidesShio.txt")
    c_mover = gnome.movers.CatsMover(curr_file, tide=Tide(shio_file))
    c_mover.scale_refpoint = (
        -70.8875, 42.321333)  # this is the value in the file (default)
    c_mover.scale = True  # default value
    c_mover.scale_value = -1
    model.movers += c_mover
    # todo: cannot add this till environment base class is created
    model.environment += c_mover.tide

    # adding a cats ossm mover

    ossm_file = os.path.join(boston_data, 'MerrimackMassCoastOSSM.txt')
    curr_file = os.path.join(boston_data, 'MerrimackMassCoast.CUR')
    c_mover = gnome.movers.CatsMover(curr_file, tide=Tide(ossm_file))
    # but do need to scale (based on river stage)
    c_mover.scale = True
    c_mover.scale_refpoint = (-70.65, 42.58333)
    c_mover.scale_value = 1.
    model.movers += c_mover
    model.environment += c_mover.tide

    # adding a cats mover

    curr_file = os.path.join(boston_data, 'MassBaySewage.CUR')
    c_mover = gnome.movers.CatsMover(curr_file)
    # but do need to scale (based on river stage)
    c_mover.scale = True
    c_mover.scale_refpoint = (-70.78333, 42.39333)
    # the scale factor is 0 if user inputs no sewage outfall effects
    c_mover.scale_value = .04
    model.movers += c_mover

    # adding a spill

    spill = WebSurfaceReleaseSpill(num_elements=1000,
                                   start_position=(
                                       -70.911432, 42.369142, 0.0),
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
