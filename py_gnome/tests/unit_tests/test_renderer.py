#!/usr/bin/env python

"""
test code for renderer

NOTE: some of the test that should be here are in test_map_canvas

"""
import os, shutil

import pytest

from gnome import renderer

bna_sample = "SampleData/MapBounds_2Spillable2Islands2Lakes.bna"
output_dir = "temp_output_dir"

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








