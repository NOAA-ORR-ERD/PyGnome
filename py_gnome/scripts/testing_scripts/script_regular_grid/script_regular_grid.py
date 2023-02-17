# A simple script for running gnome with currents on a regular grid:

import os
from pathlib import Path

import gnome.scripting as gs

# define base directory
base_dir = Path(__file__).parent


def make_model(images_dir=base_dir / 'images'):

    mapfile = base_dir / "south_colombia_coast.bna"

    start_time = "2012-4-15T00:00"

    model = gs.Model(start_time=start_time,
                     duration=gs.hours(12),
                     time_step=gs.minutes(60)
                     )

    model.map = gs.MapFromBNA(mapfile)

    curfile = gs.get_datafile(base_dir / "mareaabril.nc")
    cur_mover = gs.CurrentMover(curfile)
    model.movers += cur_mover

    # model.movers += gs.RandomMover(diffusion_coef=1e5)

    # adding a wind to make it more interesting
    model.movers += gs.constant_point_wind_mover(15, 270, "m/s")

    model.spills += gs.surface_point_line_spill(
        num_elements=10,
        amount=1.0,  # kg
        start_position=(
            -79.1,  # longitude
            1.9,  # latitude
            0.0),  # depth (0 is surface)
        end_position=(
            -79.1,  # longitude
            1.6,  # latitude
            0.0),
        release_time=start_time)

    model.outputters += gs.Renderer(mapfile,
                                    images_dir,
                                    image_size=(800, 800),
                                    viewport=((-79.25, 1.75),
                                              (-78.5, 2.25)))
    model.outputters += gs.NetCDFOutput(base_dir / "test_run.nc")

    return model


if __name__ == "__main__":
    gs.make_images_dir()
    print("setting up the model")
    model = make_model()
    print("running the model")
    model.full_run()
