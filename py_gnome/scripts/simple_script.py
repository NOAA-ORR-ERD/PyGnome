import gnome.scripting as gs

start_time = "2015-01-01"
model = gs.Model(start_time=start_time,
                 duration=gs.days(3),
                 time_step=gs.minutes(15)
                 )
                 
model.map = gs.GnomeMap(map_bounds=((-145,48), (-145,49),
                    (-143,49), (-143,48))
                    )
                    
velocity = (.2, 0, 0) #(u, v, w) in m/s
uniform_vel_mover = gs.SimpleMover(velocity)
#  random walk diffusion -- diffusion_coef in units of cm^2/s
random_mover = gs.RandomMover(diffusion_coef=2e4)

model.movers += uniform_vel_mover
model.movers += random_mover

spill = gs.surface_point_line_spill(release_time=start_time,
                                    start_position=(-144, 48.5, 0),
                                    num_elements=1000)
model.spills += spill

renderer = gs.Renderer(output_timestep=gs.hours(6),
                       map_BB=((-145,48), (-145,49),
                               (-143,49), (-143,48)))

model.outputters += renderer

model.full_run()