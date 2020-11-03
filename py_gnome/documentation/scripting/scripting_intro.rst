.. _scripting_overview:

Overview
========

To run simulations using the pyGNOME scripting environment the first step is to create a model object.
This model object will have attributes like the model start time and the run duration.

Movers, spills, and outputters are then added to the model. Then we can run the model and view the results.

For a first introduction to scripting pyGNOME, we will set up and run the very simplest model possible. We
will not load any external data, but create a simple map and movers manually. The "spill" will be a conservative
substance, i.e. it will not represent oil which has changing properties over time due to weathering processes.

The scripting module
--------------------

The ``gnome`` package contains a large hierarchy subpackages and modules that provide contain all of objects and functions that support the full functionality of the model. But unless you want to customize the algorithms, most of the functionality required is in the "scripting" module. We recommend that you import it like so::

    import gnome.scripting as gs

And then you can use most of the common objects needed for the model as so::

    gs.Model

Working with dates and times
............................

INternally, gnome uses the python standard library``datetime`` and ``timedelta`` functions. IN most cases, you can pass objects of these types inot GNOME. But for scripting convience, most places that take a datetime object will also accept a ISO 8601 string, such as: "2015-01-01T12:15:00"

gnome.scripting also provides a number of shortcuts for creating ``timedelta`` objects: ``seconds, minutes, hours, days``. You can use them ike so::

    gs.hours(3)  # for 3 hours
    gs.days(2)  # for two days


Initialize the Model
--------------------
We initialize the model to begin on New Years Day 2015 and run for 3 days::

    import gnome.scripting as gs
    start_time = "2015-01-01"
    model = gs.Model(start_time=start_time,
                     duration=gs.days(3),
                     time_step=gs.minutes(15)
                     )


Create and Add a Map
--------------------
Create a very simple map: all water with a polygon of latitude/longitude
points to specify the map bounds::


    model.map = gs.GnomeMap(map_bounds=((-145,48), (-145,49),
                                        (-143,49), (-143,48))
                                        )


Create and Add a Mover
----------------------
Now we will create some simple movers and add them to the model.
We use the SimpleMover class to specify a 0.5 m/s eastward current and
also the RandomMover class to simulate spreading due to turbulent motions::


    velocity = (.5, 0, 0) #(u, v, w) in m/s
    uniform_vel_mover = gs.SimpleMover(velocity)
    #  random walk diffusion -- diffusion_coef in units of cm^2/s
    random_mover = gs.RandomMover(diffusion_coef=2e4)

    model.movers += uniform_vel_mover
    model.movers += random_mover


Create and Add a Spill
----------------------
Spills in GNOME contain a release object which specifies the details of the release
(e.g. where, when, how many elements). They also contain an Substance object which
provides information on the type of substance spilled. For a conservative substance (i.e. one with no change in properties over time) we can use the default None value Substance.

The scripting module provides a utility function for creating a simple spill at the surface at a point or a line.
(Here we set a instantaneous spill at a point in the middle of the map)::


    spill = gs.surface_point_line_spill(release_time=start_time,
                                        start_position=(-144, 48.5, 0),
                                        num_elements=1000)
    model.spills += spill


Create and Add an Outputter
---------------------------

Outputters allow us to save our model run results. Options include saving images at specified model time steps
or saving all the particle information into a netCDF file for further analysis.

Here we use the Renderer class to save an image every 6 hours. We specify the bounding box of the rendered map to
be the same as those specified when we created the map object. The default is to save files into the working directory::


    renderer = gs.Renderer(output_timestep=timedelta(hours=6),
                           map_BB=((-145,48), (-145,49),
                                   (-143,49), (-143,48)))

    model.outputters += renderer


Step through the model and view data
------------------------------------

Once the model is all set up, we are ready to run the simulation.
Sometimes we want to do this iteratively step-by-step to view data
along the way without outputting to a file.
There are some helper utilities to extract data associated with the particles.
These data include properties such as mass, age, and position or weathering information such as the mass of oil evaporated (if the simulation has specified an oil type rather than a conservative substance as in this example).

For example, if we want to extract the particle positions as a function of time, we can use the :func:`gnome.model.get_spill_property` convenience function, as shown below::

    x=[]
    y=[]
    for step in model:
        positions = model.get_spill_property('positions')
        x.append(positions[:,0])
        y.append(positions[:,1])

To see a list of properties associated with particles use::

    model.list_spill_properties()

Note, this list will be empty until after the model has been run.


Run the model to completion
---------------------------

Alternatively, to just run the model for the entire duration use::

    model.full_run()

Results will be written to files based on the outputters added to the model.


View the results
----------------

The renderer that we added generates png images every 6 hours.
Since we did not specify an output directory for these images, they will have been saved in the same directory that the script was executed from.
The sequence of images should show a cloud of particles moving east and spreading.







