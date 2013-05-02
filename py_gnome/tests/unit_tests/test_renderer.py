#!/usr/bin/env python

"""
test code for renderer

NOTE: some of these only really test if the code crashes
  -- whether the rendering looks right -- who knows?
  -- It's a good idea to look at the output.

"""
import pytest

import os, shutil

import numpy.random as random

from gnome import basic_types
from gnome import renderer
from gnome.spill_container import TestSpillContainer

bna_sample = "SampleData/MapBounds_2Spillable2Islands2Lakes.bna"
output_dir = os.path.join(os.path.dirname(__file__), r"renderer_output_dir")
data_dir = os.path.join(os.path.dirname(__file__), r"SampleData")

if os.path.isdir(output_dir):
    shutil.rmtree(output_dir)
os.mkdir(output_dir)


def test_init():
    
    r = renderer.Renderer(bna_sample,
                          output_dir)

    assert True

def test_file_delete():
    
    r = renderer.Renderer(bna_sample,
                          output_dir)
    # dump some files into output dir:
    open(os.path.join(output_dir, r.background_map_name),'w').write("some junk")

    for i in range(5):
        open(os.path.join(output_dir, r.foreground_filename_format%i),'w').write("some junk")


    r.prepare_for_model_run()

    # there should only be a background image now.
    files = os.listdir(output_dir)
    assert files == [r.background_map_name]

def test_render_elements():
    """
    Should this test be in map_cnavas?
    """
    r = renderer.Renderer(os.path.join(data_dir, 'MapBounds_2Spillable2Islands2Lakes.bna'),
                          output_dir,
                          image_size=(800,600))
    BB = r.map_BB
    min_lon, min_lat = BB[0] 
    max_lon, max_lat = BB[1] 
    
    N = 1000
    #create some random particle positions:
    lon = random.uniform(min_lon, max_lon, (N,))
    lat = random.uniform(min_lat, max_lat, (N,))

    #create a spill
    spill = TestSpillContainer(num_elements=N)
    spill['positions'][:,0] = lon
    spill['positions'][:,1] = lat

    r.create_foreground_image()
    r.draw_elements(spill)

    # create an uncertainty spill
    lon = random.uniform(min_lon, max_lon, (N,))
    lat = random.uniform(min_lat, max_lat, (N,))

    spill = TestSpillContainer(num_elements=N, uncertain=True)
    spill['positions'][:,0] = lon
    spill['positions'][:,1] = lat

    r.draw_elements(spill)

    # save the image
    r.save_foreground( os.path.join(output_dir,"foreground1.png") )
    assert True        

def test_render_beached_elements():
    """
    Should this test be in map_cnavas?
    """
    r = renderer.Renderer(os.path.join(data_dir, 'MapBounds_2Spillable2Islands2Lakes.bna'),
                          output_dir,
                          image_size=(800,600))
    BB = r.map_BB
    min_lon, min_lat = BB[0] 
    max_lon, max_lat = BB[1] 
    
    N = 100
    # create some random particle positions:
    lon = random.uniform(min_lon, max_lon, (N,))
    lat = random.uniform(min_lat, max_lat, (N,))

    # create a spill
    spill = TestSpillContainer(num_elements=N)
    spill['positions'][:,0] = lon
    spill['positions'][:,1] = lat
    # make half of them on land
    spill['status_codes'][::2]   = basic_types.oil_status.on_land

    r.create_foreground_image()
    r.draw_elements(spill)

#    assert False

    # create an uncertainty spill
    lon = random.uniform(min_lon, max_lon, (N,))
    lat = random.uniform(min_lat, max_lat, (N,))

    spill = TestSpillContainer(num_elements=N, uncertain=True)
    spill['positions'][:,0] = lon
    spill['positions'][:,1] = lat
    # make half of them on land
    spill['status_codes'][::2]   = basic_types.oil_status.on_land

    r.draw_elements(spill)

    # save the image
    r.save_foreground( os.path.join(output_dir,"foreground2.png") )
    assert True        


if __name__ == "__main__":
    pass





