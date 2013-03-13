import json
import os
import datetime
import gnome.basic_types
import numpy
import sys

from pyramid.paster import bootstrap
from webgnome import WebSurfaceReleaseSpill, WebWindMover, util
from webgnome.model_manager import WebRandomMover, WebWind
from webgnome.schema import ModelSchema


def main():
    """
    Configure a model with the parameters of the Long Island script.
    """
    env = bootstrap('../development.ini')
    settings = env['registry'].settings
    location_file_dir = settings.location_file_dir
    location_file = os.path.join(location_file_dir, 'long_island',
                                 'location.json')

    if os.path.exists(location_file):
        print >> sys.stderr, 'File already exists: %s' % location_file
        exit(1)

    model = settings.Model.create()

    spill = WebSurfaceReleaseSpill(
        name="Long Island Spill",
        num_elements=1000,
        start_position=(-72.419992, 41.202120, 0.0),
        release_time=model.start_time)

    model.spills.add(spill)

    start_time = model.start_time

    r_mover = WebRandomMover(diffusion_coef=500000)
    model.movers.add(r_mover)

    series = numpy.zeros((5,), dtype=gnome.basic_types.datetime_value_2d)
    series[0] = (start_time, (30, 50))
    series[1] = (start_time + datetime.timedelta(hours=18), (30, 50))
    series[2] = (start_time + datetime.timedelta(hours=30), (20, 25))
    series[3] = (start_time + datetime.timedelta(hours=42), (25, 10))
    series[4] = (start_time + datetime.timedelta(hours=54), (25, 180))

    wind = WebWind(units='mps', timeseries=series)
    w_mover = WebWindMover(wind=wind, is_constant=False)
    model.movers.add(w_mover)

    map_file = os.path.join(
        'location_files', 'long_island', 'data', 'LongIslandSoundMap.BNA')

    model.add_bna_map(map_file, {
        'refloat_halflife': 6 * 3600,
        'name': "Long Island Sound"
    })

    model.uncertain = False

    serialized_model = ModelSchema().bind().serialize(model.to_dict())
    model_json = json.dumps(serialized_model, default=util.json_encoder,
                            indent=4)

    with open(location_file, 'wb') as f:
        f.write(model_json)

    env['closer']()

if __name__ == '__main__':
    main()