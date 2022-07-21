#!/usr/bin/env python

"""
tests for the tidal flat map
"""





# a few tests that show that the delation to the underlying map works:

import os
from datetime import datetime
import numpy as np


from gnome.maps import GnomeMap, MapFromBNA
from gnome.basic_types import oil_status

from gnome.maps.tideflat_map import (TideflatMap,
                                     TideflatBase,
                                     SimpleTideflat,
                                     )
import gnome.scripting as gs

import pytest

this_dir = os.path.dirname(__file__)

output_dir = os.path.join(this_dir, "output")

bna_file = os.path.join(this_dir, "small_wadden_sea.bna")


# this should go in conftest somewhere -- there is a dump_folder
#  but it's session level -- I prefer module-level
@pytest.fixture(scope="module")
def output_dir():
    '''
    create dir for output data/files
    module scope so it is only executed the first time it is used

    '''
    # fixme: this SHOULD use pytest fixtures ability to know what module
    # its running it to create the dir

    this_dir = os.path.dirname(__file__)
    dump_loc = os.path.join(this_dir, "output_tideflat_map")

    try:
        shutil.rmtree(dump_loc)
    #fixme: what Exception??
    except Exception:
        pass
    try:
        os.makedirs(dump_loc)
    except Exception:
        pass
    return dump_loc



@pytest.fixture
def simple_model(output_dir):
    start_time = "2018-09-20T12:00"
    model = gs.Model(start_time=start_time,
                     duration=gs.days(1),
                     time_step=gs.minutes(30),
                     name="test model for tideflats",
                     )
    model.map = MapFromBNA(bna_file)
    model.movers += gs.constant_point_wind_mover(10, 300, "m/s")

    model.spills += gs.surface_point_line_spill(num_elements=100,
                                                start_position=(5.4, 53.38, 0),
                                                end_position=(5.8, 53.4, 0),
                                                release_time=start_time,
                                                )
    model.outputters += gs.Renderer(output_timestep=gs.hours(1),
                                    map_filename=bna_file,
                                    output_dir=output_dir,
                                    formats=['gif'],  # ['gif', 'png']
                                    image_size=(800, 400),
                                    # viewport=((4.5, 53.0),
                                    #           (5.5, 53.5)),
                                    )

    return model


def get_gnomemap():
    return GnomeMap(map_bounds=((10, 10),
                                (15, 10),
                                (15, 15),
                                (10, 15)),
                    spillable_area=((11, 11),
                                    (14, 11),
                                    (14, 14),
                                    (11, 14)),

                    land_polys=None,
                    name="Test Map for TideflatMap"
                    )


def get_simple_tideflat():
    dry_start = datetime(2018, 1, 1, 12)
    dry_end = datetime(2018, 1, 1, 13)
    bounds = ((12, 12),
              (13, 12),
              (13, 13),
              (12, 13),
              )
    return SimpleTideflat(bounds, dry_start, dry_end)


def test_with_gnome_map():
    """
    simplest possible -- the all water map
    """
    # this should now work like a regular map
    tfm = TideflatMap(get_gnomemap(), None)

    # a few things to make sure they work:
    assert tfm.on_map((11, 11, 0))
    assert not tfm.on_map((16, 16, 0))

    assert tfm.in_water((11, 11, 0))

    assert tfm.allowable_spill_position((13, 13, 0))

    assert not tfm.allowable_spill_position((10.5, 10.5, 0))

def test_TideFlatBase():
    """
    doesn't do much but should be able to be created
    """
    tf = TideflatBase()
    # any points should be never dry
    points = ((0, 0, 0), (100, 100, 0))
    result = tf.is_dry(points, datetime(2018, 1, 1, 0))
    assert np.all(result == False)


def test_SimpleTideFlat():
    """
    testing the simple tide flat implimentaiton

    so it can then be used for tests...
    """
    tf = get_simple_tideflat()
    # outside the bounds of time -- where doesn't matter
    assert not tf.is_dry((3, 4, 5), datetime(2018, 1, 1, 11))[0]
    assert not tf.is_dry((3, 4, 5), datetime(2018, 1, 1, 14))[0]

    # time in-bounds:
    dt = datetime(2018, 1, 1, 12, 30)

    # one point, one out
    points = ((12.5, 12.5, 0),  # in
              (12, 11.5, 0))    # out
    result = tf.is_dry(points, dt)
    assert np.all(result == [True, False])

    result = tf.is_wet(points, dt)
    assert np.all(result == [False, True])



def test_tideflat_map_with_both():
    tfm = TideflatMap(get_gnomemap(), get_simple_tideflat())

    # a few things to make sure they work:
    assert tfm.on_map((11, 11, 0))
    assert not tfm.on_map((16, 16, 0))

    assert tfm.in_water((11, 11, 0))
    assert tfm.allowable_spill_position((13, 13, 0))

# now the real stuff!


def test_refloat_elements():
    tfm = TideflatMap(get_gnomemap(), get_simple_tideflat())

    # Fake spill_container
    sc = {'next_positions': np.array(((12, 11, 0),  # in water
                                      (13.5, 13.5, 0),  # on_land (if the map did that)
                                      (12.5, 12.5, 0),  # still on tideflat
                                      (11.5, 12.5, 0),  # no longer on tideflat
                                      (12.5, 13.5, 0),  # in water
                                      )),
          'status_codes': np.array((oil_status.in_water,
                                    oil_status.on_land,
                                    oil_status.on_tideflat,
                                    oil_status.on_tideflat,
                                    oil_status.in_water,
                                    ))}

    tfm.refloat_elements(sc, gs.minutes(10), datetime(2018, 1, 1, 12, 30))
    assert np.all(sc['status_codes'] == np.array((oil_status.in_water,
                                                  oil_status.on_land,
                                                  oil_status.on_tideflat,
                                                  oil_status.in_water,
                                                  oil_status.in_water,
                                                  ))
                  )


def test_full_model_run(simple_model):
    """
    run it with a full model and no tide flats

    no tests here, but you can look at the output
    """
    # run it a bit faster
    # but long enough for them all to beach
    simple_model.duration = gs.hours(24)
    # simple_model.full_run()
    for step in simple_model:
        print("step num:", step['step_num'])

    status = simple_model.get_spill_property('status_codes')

    assert np.all(status == oil_status.on_land)


def test_model_run_with_tideflat(simple_model):
    """
    Add a tideflat with the simple tideflat object

    no tests here, but you can look at the output
    """
    model = simple_model

    # make a simple tideflat model
    bounds = ((5.623211, 53.309485),
              (5.784850, 53.348716),
              (5.761970, 53.368978),
              (5.722114, 53.376904),
              (5.667496, 53.367657),
              (5.620259, 53.354003),
              (5.609926, 53.328444),
              )

    dry_start = model.start_time + gs.hours(4)
    dry_end = model.start_time + gs.hours(8)

    tf = SimpleTideflat(bounds, dry_start, dry_end)

    # get the map from the model and wrap it in a TideflatMap
    tfm = TideflatMap(model.map, tf)

    model.map = tfm

    # to make it run faster
    model.time_step = gs.hours(2)
    for step in model:
        print("step_num", step['step_num'])

    status = model.get_spill_property('status_codes')

    assert np.all(status == oil_status.on_land)


