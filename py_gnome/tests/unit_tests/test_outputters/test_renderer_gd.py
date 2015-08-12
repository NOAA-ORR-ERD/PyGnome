#!/usr/bin/env python

"""
test code for renderer

-- this version uses teh map_canvas_gd (uses libgd)

NOTE: some of these only really test if the code crashes
  -- whether the rendering looks right -- who knows?
  -- It's a good idea to look at the output.

"""

import os
from datetime import datetime

import pytest
import numpy.random as random

from gnome.basic_types import oil_status

from gnome.outputters.renderer_gd import Renderer
from gnome.utilities.projections import GeoProjection

from ..conftest import sample_sc_release, testdata


bna_sample = testdata['Renderer']['bna_sample']
bna_star = testdata['Renderer']['bna_star']

# ## fixme: put this in a fixture?
# output_dir = os.path.join(os.path.split(__file__)[0], "sample_output")

# try:
#     os.mkdir(output_dir)
# except OSError:
#     pass # already there


def test_exception(output_dir):
    # wrong name for draw on top
    with pytest.raises(ValueError):
        Renderer(bna_sample, output_dir, draw_ontop='forecasting')

    r = Renderer(bna_sample, output_dir)
    with pytest.raises(ValueError):
        r.draw_ontop = 'forecasting'


def test_init(output_dir):
    r = Renderer(bna_sample, output_dir)
    print r

    assert True


def test_file_delete(output_dir):

    output_dir = os.path.join(output_dir, 'clear_test')

    r = Renderer(bna_sample, output_dir)
    bg_name = r.background_map_name
    fg_format = r.foreground_filename_format

    # dump some files into output dir:

    open(os.path.join(output_dir, bg_name), 'w').write('some junk')

    for i in range(5):
        open(os.path.join(output_dir, fg_format.format(i)), 'w'
             ).write('some junk')

    r.prepare_for_model_run(model_start_time=datetime.now())

    # there should only be a background image now.

    files = os.listdir(output_dir)
    assert files == [r.background_map_name]


def test_rewind(output_dir):
    'test rewind calls base function and clear_output_dir'
    output_dir = os.path.join(output_dir, 'clear_test')
    r = Renderer(bna_sample, output_dir)
    bg_name = r.background_map_name
    fg_format = r.foreground_filename_format

    # dump some files into output dir:

    open(os.path.join(output_dir, bg_name), 'w').write('some junk')

    for i in range(5):
        open(os.path.join(output_dir,
             fg_format.format(i)),
             'w'
             ).write('some junk')

    now = datetime.now()
    r.prepare_for_model_run(model_start_time=now)

    assert r._model_start_time == now

    # prepare for model run clears output dir, but adds in the background map
    files = os.listdir(output_dir)
    assert files == [r.background_map_name]

    r.rewind()

    assert r._model_start_time is None  # check superclass rewind is called

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

    r = Renderer(bna_star, output_dir, image_size=(600, 600),
                 projection_class=GeoProjection)

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


@pytest.mark.parametrize(("json_"), ['save', 'webapi'])
def test_serialize_deserialize(json_, output_dir):
    r = Renderer(bna_sample, output_dir)
    r2 = Renderer.new_from_dict(r.deserialize(r.serialize(json_)))
    if json_ == 'save':
        assert r == r2


# # if __name__ == '__main__':
# #     test_set_viewport()
