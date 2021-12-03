#!/usr/bin/env python

"""
test code for renderer

-- this version uses the map_canvas_gd (uses libgd)

NOTE: some of these only really test if the code crashes
  -- whether the rendering looks right -- who knows?
  -- It's a good idea to look at the output.

"""

import os
from os.path import basename

import numpy as np

from datetime import datetime

import pytest
import numpy.random as random

from gnome.basic_types import oil_status

from gnome.outputters.renderer import Renderer
from gnome.utilities.projections import GeoProjection

from ..conftest import sample_sc_release, testdata

import gnome.scripting as gs

# fixme -- this should be in conftest
from gnome.spill_container import SpillContainerPairData

bna_sample = testdata['Renderer']['bna_sample']
bna_star = testdata['Renderer']['bna_star']


class FakeCache(object):
    def __init__(self, sc):
        # pass in a  spill container
        self.sc = sc
        self.sc.current_time_stamp = datetime.now()

    def load_timestep(self, step):
        print(f"loading step {step}")
        return SpillContainerPairData(self.sc)


def test_exception(output_dir):
    # wrong name for draw on top
    with pytest.raises(ValueError):
        Renderer(bna_sample, output_dir, draw_ontop='forecasting')

    r = Renderer(bna_sample, output_dir)
    with pytest.raises(ValueError):
        r.draw_ontop = 'forecasting'


def test_init(output_dir):
    r = Renderer(bna_sample, output_dir)

    assert True


def test_file_delete(output_dir):

    output_dir = os.path.join(output_dir, 'clear_test')

    r = Renderer(bna_sample, output_dir)
    bg_name = r.background_map_name + "png"
    fg_format = r.foreground_filename_format

    # dump some files into output dir:
    open(os.path.join(output_dir, bg_name), 'w').write('some junk')

    for i in range(5):
        open(os.path.join(output_dir, fg_format.format(i) + "png"), 'w'
             ).write('some junk')

    r.prepare_for_model_run(model_start_time=datetime.now())

    # The default output formats are ['png','gif']
    # so now there should only be a background image and the animated gif.
    files = sorted(os.listdir(output_dir))
    assert files == sorted([os.path.basename(r.anim_filename),
                            r.background_map_name + "png"])


def test_rewind(output_dir):
    'test rewind calls base function and clear_output_dir'
    output_dir = os.path.join(output_dir, 'clear_test')
    r = Renderer(bna_sample, output_dir)
    bg_name = r.background_map_name
    fg_format = r.foreground_filename_format

    # dump some files into output dir:
    open(os.path.join(output_dir, bg_name) + 'png', 'w').write('some junk')

    for i in range(5):
        open(os.path.join(output_dir,
             fg_format.format(i) + 'png'),
             'w'
             ).write('some junk')

    now = datetime.now()
    r.prepare_for_model_run(model_start_time=now)

    assert r._model_start_time == now

    # prepare for model run clears output dir, but adds in the background map
    files = sorted(os.listdir(output_dir))
    # using defaults, so we know the file extension
    assert files == sorted([os.path.basename(r.anim_filename),
                            r.background_map_name + 'png'])

    r.rewind()

    # check super is called correctly
    assert r._model_start_time is None
    assert r._dt_since_lastoutput is None
    assert r._write_step is True

    # changed renderer and netcdf ouputter to delete old files in
    # prepare_for_model_run() rather than rewind()
    # -- rewind() was getting called a lot
    # -- before there was time to change the output file names, etc.
    # So for this unit test, there should only be a background image now.
    files = sorted(os.listdir(output_dir))
    assert files == sorted([os.path.basename(r.anim_filename),
                            r.background_map_name + 'png'])

def test_render_basemap(output_dir):
    """
    render the basemap
    """
    r = Renderer(bna_star, output_dir, image_size=(600, 600))

    r.draw_background()
    r.save_background(os.path.join(output_dir, 'basemap.png'))


def test_render_basemap_with_bounds(output_dir):
    """
    render the basemap
    """
    r = Renderer(bna_sample, output_dir, image_size=(600, 600))

    r.draw_background()
    r.save_background(os.path.join(output_dir, 'basemap_bounds.png'))


def test_render_elements(output_dir):
    """
    See if the "splots" get rendered corectly
    """

    r = Renderer(bna_sample, output_dir, image_size=(400, 400))

    BB = r.map_BB
    (min_lon, min_lat) = BB[0]
    (max_lon, max_lat) = BB[1]

    N = 1000

    # create some random particle positions:

    lon = random.uniform(min_lon, max_lon, (N, ))
    lat = random.uniform(min_lat, max_lat, (N, ))

    # create a sc

    sc = sample_sc_release(num_elements=N)
    sc['positions'][:, 0] = lon
    sc['positions'][:, 1] = lat

    r.draw_elements(sc)

    # create an uncertainty sc

    lon = random.uniform(min_lon, max_lon, (N, ))
    lat = random.uniform(min_lat, max_lat, (N, ))

    sc = sample_sc_release(num_elements=N, uncertain=True)
    sc['positions'][:, 0] = lon
    sc['positions'][:, 1] = lat

    r.draw_elements(sc)

    # save the image

    r.save_foreground(os.path.join(output_dir, 'elements1.png'))


def test_write_output(output_dir):
    """
    render the basemap
    """
    r = Renderer(bna_star,
                 output_dir,
                 image_size=(600, 600),
                 draw_back_to_fore=True,
                 formats=['png'])

    r.draw_background()

    BB = r.map_BB
    (min_lon, min_lat) = BB[0]
    (max_lon, max_lat) = BB[1]

    N = 100
    # create some random particle positions:
    lon = random.uniform(min_lon, max_lon, (N, ))
    lat = random.uniform(min_lat, max_lat, (N, ))

    # create a sc
    sc = sample_sc_release(num_elements=N)
    sc['positions'][:, 0] = lon
    sc['positions'][:, 1] = lat

    r.cache = FakeCache(sc)

    r.write_output(0)
    r.save_foreground(os.path.join(output_dir, 'map_and_elements.png'))

    r.draw_back_to_fore = False
    r.clear_foreground()
    r.write_output(1)
    r.save_foreground(os.path.join(output_dir, 'just_elements.png'))


def test_render_beached_elements(output_dir):

    r = Renderer(bna_sample,
                 output_dir,
                 image_size=(800, 600))

    BB = r.map_BB
    (min_lon, min_lat) = BB[0]
    (max_lon, max_lat) = BB[1]

    N = 100

    # create some random particle positions:

    lon = random.uniform(min_lon, max_lon, (N, ))
    lat = random.uniform(min_lat, max_lat, (N, ))

    # create a sc

    sc = sample_sc_release(num_elements=N)
    sc['positions'][:, 0] = lon
    sc['positions'][:, 1] = lat

    # make half of them on land

    sc['status_codes'][::2] = oil_status.on_land

    r.draw_elements(sc)

    # create an uncertainty sc

    lon = random.uniform(min_lon, max_lon, (N, ))
    lat = random.uniform(min_lat, max_lat, (N, ))

    sc = sample_sc_release(num_elements=N, uncertain=True)
    sc['positions'][:, 0] = lon
    sc['positions'][:, 1] = lat

    # make half of them on land

    sc['status_codes'][::2] = oil_status.on_land

    r.draw_elements(sc)

    # save the image

    r.save_foreground(os.path.join(output_dir, 'elements2.png'))


def test_show_hide_map_bounds(output_dir):
    r = Renderer(bna_star, output_dir, image_size=(600, 600))

    r.draw_background()
    r.save_background(os.path.join(output_dir, 'star_background.png'))

    # try again without the map bounds:

    r.draw_map_bounds = False
    r.draw_background()
    r.save_background(os.path.join(output_dir,
                      'star_background_no_bound.png'))


def test_set_viewport(output_dir):
    """
    tests various rendering, re-zooming, etc

    NOTE: this will only test if the code crashes, you have to look
          at the rendered images to see if it does the right thing
    """

    r = Renderer(bna_star,
                 output_dir,
                 image_size=(600, 600),
                 projection=GeoProjection(),
                 )

    # re-scale:
    # should show upper right corner

    r.viewport = ((-73, 40), (-70, 43))
    r.draw_background()
    r.save_background(os.path.join(output_dir, 'star_upper_right.png'))

    # re-scale:
    # should show lower right corner

    r.viewport = ((-73, 37), (-70, 40))
    r.draw_background()
    r.save_background(os.path.join(output_dir, 'star_lower_right.png'))

    # re-scale:
    # should show lower left corner

    r.viewport = ((-76, 37), (-73, 40))
    r.draw_background()
    r.save_background(os.path.join(output_dir, 'star_lower_left.png'))

    # re-scale:
    # should show upper left corner

    r.viewport = ((-76, 40), (-73, 43))
    r.draw_background()
    r.save_background(os.path.join(output_dir, 'star_upper_left.png'))


def test_draw_raster_map(output_dir):
    """
    tests drawing the raster map

    Note: you need to look at the output to know if it did it right...

    """
    import gnome

    r = Renderer(bna_sample, image_size=(1000, 1000))
    r.viewport = ((-127.47, 48.10), (-127.22, 48.24))

    r.draw_background()

    # make a raster map out of the BNA:
    r.raster_map = gnome.map.MapFromBNA(bna_sample,
                                        raster_size=10000)

    r.raster_map_outline = True
    r.draw_raster_map()

    r.save_background(os.path.join(output_dir, 'raster_map_render.png'))


def test_serialize_deserialize(output_dir):
    # non-defaults to check properly..
    r = Renderer(map_filename=bna_sample,
                 output_dir=output_dir,
                 image_size=(1000, 800),
                 viewport=((-126.5, 47.5),
                           (-126.0, 48.0)))

    toserial = r.serialize()

    r2 = Renderer.deserialize(toserial)

    assert r == r2

def fake_run_for_animation(rend):

    # create a sc
    sc = sample_sc_release(num_elements=100)

    lon = np.random.random((100,))
    lat = np.random.random((100,))

    sc['positions'][:, 0] = lon
    sc['positions'][:, 1] = lat

    rend.cache = FakeCache(sc)

    stime = datetime(2021, 1, 1, 0)
    rend.prepare_for_model_run(model_start_time=stime)
    for step in range(30):
        rend.prepare_for_model_step(3600, stime)
        rend.write_output(step, islast_step=False)
        # move the elements
        sc['positions'] += 0.02
    rend.post_model_run()


def test_animation(output_dir):
    """
    tests the animated gif support
    """
    odir = os.path.join(output_dir, "animation")
    rend = Renderer(output_dir=odir,
                    image_size=(400, 400),
                    viewport=(((0, 0),(1, 1))),
                    formats=['gif']
                    )
    fake_run_for_animation(rend)

    # do it again: make sure it got cleaned up
    fake_run_for_animation(rend)

def test_animation_in_model(output_dir):
    """
    note: this is probably not the least bit necessary, but there you go
    """
    model = gs.Model()
    model.movers += gs.RandomMover()
    model.spills += gs.surface_point_line_spill(num_elements=100,
                                                start_position=(0, 0),
                                                release_time=model.start_time,
                                                )
    odir = os.path.join(output_dir, "animation_model")
    model.outputters += Renderer(output_dir=odir,
                                 image_size=(400, 400),
                                 viewport=(((-0.02, -0.02), (0.02, 0.02))),
                                 formats=['gif']
                                 )

    model.full_run()

#    assert False

# # if __name__ == '__main__':
# #     test_set_viewport()
