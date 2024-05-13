"""
simple_script.py

About the simplest script you can write

This example is documented in the PYGNOME docs: Scripting Overview
"""

# easiest way to get all the common functionality in one place
import gnome.scripting as gs
# gnome can use either ISO style strings or datetime objects
start_time = "2015-01-01"

model = gs.Model(start_time=start_time,
                 duration=gs.days(3),
                 time_step=gs.minutes(15)
                 )

# The base GnomeMap is all water, no land
# you can optionally add boundaries
model.map = gs.GnomeMap(map_bounds=((-145, 48), (-145, 49),
                                    (-143, 49), (-143, 48))
                        )


# The very simplest mover: a steady uniform current
velocity = (.2, 0, 0)  # (u, v, w) in m/s
uniform_vel_mover = gs.SimpleMover(velocity)


#  random walk diffusion -- diffusion_coef in units of cm^2/s
random_mover = gs.RandomMover(diffusion_coef=2e4)

# add the movers to the model
model.movers += uniform_vel_mover
model.movers += random_mover

# create spill
spill = gs.surface_point_line_spill(release_time=start_time,
                                    start_position=(-144, 48.5),
                                    num_elements=1000)
# add it to the model
model.spills += spill

# create an outputter: this renders png files and an animated gif
renderer = gs.Renderer(output_dir='./output/',
                       output_timestep=gs.hours(2),
                       # bounding box for the output images
                       viewport=((-145, 48), (-145, 49),
                                 (-143, 49), (-143, 48)),
                       formats=['gif']
                       )

model.outputters += renderer

print("running the model: see output in the output dir")
# run the model
model.full_run()
