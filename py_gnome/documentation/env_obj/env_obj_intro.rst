.. _env_obj_overview:

Overview
========
To run simulations using the pyGNOME scripting environment the first step is to create a model object.
This model object will have attributes like the model start time and the run duration.

Movers, spills, and outputters are then added to the model. Then we can run the model and view the results.

For a first introduction to scripting pyGNOME, we will set up and run the very simplest model possible. We
will not load any external data, but create a simple map and movers manually. The "spill" will be a conservative
substance, i.e. it will not represent oil which has changing properties over time due to weathering processes.

Initialize the Model
--------------------
We initialize the model to begin on New Years Day 2015 and run for 3 days::

    from gnome.model import Model
    from datetime import datetime, timedelta
    start_time = datetime(2015, 1, 1, 0, 0)
    model = Model(start_time=start_time,
                  duration=timedelta(days=3),
                  time_step=60 * 15, #seconds
                  )

Create and Add a Map
--------------------
Create a very simple map: all water with a polygon of latitude/longitude
points to specify the map bounds::

    from gnome.map import GnomeMap
    model.map = GnomeMap(map_bounds=((-145,48), (-145,49), (-143,49), (-143,48)))

Create and Add a Mover
----------------------
Now we will create some simple movers and add them to the model.
We use the SimpleMover class to specify a 0.5 m/s eastward current and
also the RandomMover class to simulate spreading due to turbulent motions::

    from gnome.movers import SimpleMover, RandomMover
    velocity = (.5,0,0) #(u,v,w) in m/s
    uniform_vel_mover = SimpleMover(velocity)
    random_mover = RandomMover()

    model.movers += uniform_vel_mover
    model.movers += random_mover

Create and Add a Spill
----------------------
Spills in GNOME contain a release object which specifies the details of the release
(e.g. where, when, how many elements). They also contain an element_type object which
provides information on the type of substance spilled. For a conservative substance (i.e. one with
no change in properties over time) we can use the default None value for element_type.

We'll use a simple release class which can be used for an instantaneous or continuous release which can
occur at a single point, or over a line.
(Here we set a instantaneous spill at a point in the middle of the map)::

    from gnome.spill import PointLineRelease, Spill
    release = PointLineRelease(release_time=start_time,start_position=(-144,48.5,0),num_elements=1000)
    spill = Spill(release)
    model.spills += spill


Create and Add an Outputter
---------------------------
Outputters allow us to save our model run results. Options include saving images at specified model time steps
or saving all the particle information into a netCDF file for further analysis.

Here we use the Renderer class to save an image every 6 hours. We specify the bounding box of the rendered map to
be the same as those specified when we created the map object. The default is to save files into the working directory::

    from gnome.outputters import Renderer
    renderer = Renderer(output_timestep=timedelta(hours=6),map_BB=((-145,48), (-145,49), (-143,49), (-143,48)))

    model.outputters += renderer

Run the model
-------------
The model can be run by stepping through individual timesteps (e.g. if we want to see results along the way) or we
can do a full run::

    model.full_run()

View the results
----------------
The renderer that we added generated png images every 6 hours. Since we did not specify an output directory for these images,
they will have been saved in the same directory that the script was executed from. The sequence of images should show a cloud
of particles moving east and spreading.






