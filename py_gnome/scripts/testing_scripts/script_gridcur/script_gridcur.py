#!/usr/bin/env python

"""
script for demonstrating working with the gridcur forma

gridcur is an ASCII format originally used for GNOME 1
"""

from pathlib import Path
from gnome import scripting as gs

gs.set_verbose()

# define base directory
base_dir = Path(__file__).parent

def make_model(images_dir=base_dir / "images"):

    start_time = "2020-07-14T12:00"

    model = gs.Model(time_step=gs.hours(1),
                     start_time=start_time,
                     duration=gs.hours(12),
                     uncertain=False)

    current = gs.FileGridCurrent(base_dir / "example_gridcur_on_nodes.cur")

    mover = gs.CurrentMover(current=current)

    model.movers += mover

    spill = gs.grid_spill(bounds=((-88.0, 29.0),
                                  (-86.0, 30.0),
                                  ),
                          resolution=20,
                          release_time=start_time,
                          )
    model.spills += spill
    renderer = gs.Renderer(output_dir=images_dir,
                           image_size=(800, 600),
                           viewport=((-88.0, 29.0),
                                     (-86.0, 30.0),
                                     ),
                           )
    model.outputters += renderer

    return model

if __name__ == "__main__":
    gs.make_images_dir()
    print("setting up the model")
    model = make_model()
    print("running the model")
    model.full_run()
