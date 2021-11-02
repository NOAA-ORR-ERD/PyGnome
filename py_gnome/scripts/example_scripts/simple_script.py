"""
simple_script.py

About the simplest script you can write
"""

# easiest way to get all the common functionality in one place
import gnome.scripting as gs

start_time = "2015-01-01"  # gnome can use either ISO style strings or datetime objects

model = gs.Model(start_time=start_time,
                 duration=gs.days(3),
                 time_step=gs.minutes(15)
                 )

# the base GnomeMap is all water, no land
# you can optionally add boundaries
model.map = gs.GnomeMap(map_bounds=((-145, 48), (-145, 49),
                                    (-143, 49), (-143, 48))
                        )


# The very simplest mover: a steady uniform current
velocity = (.2, 0, 0) #(u, v, w) in m/s
uniform_vel_mover = gs.SimpleMover(velocity)


#  random walk diffusion -- diffusion_coef in units of cm^2/s
random_mover = gs.RandomMover(diffusion_coef=2e4)

# add the movers to the model
model.movers += uniform_vel_mover
model.movers += random_mover

# create spill
spill = gs.surface_point_line_spill(release_time=start_time,
                                    start_position=(-144, 48.5, 0),
                                    num_elements=1000)
# add it to the model
model.spills += spill

# create an outputter: this renders png files and an animated gif
renderer = gs.Renderer(output_dir='./output/',
                       output_timestep=gs.hours(6),
                       # bounding box for the output images
                       map_BB=((-145, 48), (-145, 49),
                               (-143, 49), (-143, 48)))

model.outputters += renderer

# run the model
model.full_run()