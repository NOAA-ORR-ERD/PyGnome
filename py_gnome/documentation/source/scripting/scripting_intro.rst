.. _scripting_overview:

Overview
========

To run simulations using the ``gnome`` package the first step is to create a model object.
This model object will have attributes like the model start time and the run duration. Movers, weatherers, spills, 
and outputters are then added to the model. Then the model can be run to produce output.

For a first introduction to scripting with ``gnome``, the example below will demonstrate how to set up and run the 
very simplest model possible. 
This example does load any external data, but creates a simple map and movers manually. The "spill" will be a conservative
substance, i.e. it will not represent oil which has changing properties over time due to weathering processes.

The scripting module
--------------------

The ``gnome`` package contains a large hierarchy of subpackages and modules that contain all of objects and functions that support the full functionality of the model. Therefore, creating objects and adding them to the model can involve importing classes from many different modules. To make this easier, all of the typically required classes and helper functions are made accessible via the "scripting" module. We recommend that you import it like so::

    import gnome.scripting as gs

And then you can use most of the common objects needed for the model. For example::

    model = gs.Model()
    
This is equivalent to::
    
    import gnome
    model = gnome.model.Model()

.. admonition:: Working with dates and times

    Internally, ``gnome`` uses the python standard library ``datetime`` and ``timedelta`` functions. In most cases, you can pass objects of these types into ``gnome`` . But for scripting convience, most places that take a datetime object will also accept a ISO 8601 string, such as: "2015-01-01T12:15:00"

The gnome.scripting module also provides a number of shortcuts for creating ``timedelta`` objects: ``seconds, minutes, hours, days``. You can use them like so::

    gs.hours(3)  # for 3 hours
    gs.days(2)  # for two days

Examples in this section will use the scripting module. In the following more detailed sections about specific object types, we may sometimes also show the full import path for clarity.

Initialize the Model
--------------------
We initialize the model to begin on New Years Day 2015 and run for 3 days with a model time step of 3 minutes::

    import gnome.scripting as gs
    start_time = "2015-01-01"
    model = gs.Model(start_time=start_time,
                     duration=gs.days(3),
                     time_step=gs.minutes(15)
                     )


Create and Add a Map
--------------------
Create a very simple map whis is all water with a polygon of latitude/longitude
points to specify the map bounds::

    model.map = gs.GnomeMap(map_bounds=((-145,48), (-145,49),
                                        (-143,49), (-143,48))
                                        )

Create and Add a Mover
----------------------
Now we will create some simple movers and add them to the model.
We use the :class:`gnome.movers.SimpleMover` class to specify a 0.2 m/s eastward current and
also the :class:`gnome.movers.RandomMover` class to simulate spreading due to turbulent motions::

    velocity = (.2, 0, 0) #(u, v, w) in m/s
    uniform_vel_mover = gs.SimpleMover(velocity)
    #  random walk diffusion -- diffusion_coef in units of cm^2/s
    random_mover = gs.RandomMover(diffusion_coef=2e4)

    model.movers += uniform_vel_mover
    model.movers += random_mover


Create and Add a Spill
----------------------
Spills in ``gnome`` contain information about the release (where, when, how much) in a Release Object and information about 
the properties of the substance spilled (e.g. oil chemistry) in a Substance Object. There are a number of "helper" functions 
to make it easier to initialize various types of spills (for example, at a point or over a spatial 
area, at the surface or subsurface).
 
Here we use the :func:`gnome.spill.spill.surface_point_line_spill` function to initialize a simple spill of a conservative substance 
(i.e. one with no change in properties over time) at a single point on the ocean surface::


    spill = gs.surface_point_line_spill(release_time=start_time,
                                        start_position=(-144, 48.5, 0),
                                        num_elements=1000)
    model.spills += spill


Create and Add an Outputter
---------------------------

Outputters allow us to save our model run results. Options include saving images at specified model time steps
or saving all the particle information into a netCDF file for further analysis.

Here we use the :class:`gnome.outputters.Renderer` class to save an image every 6 hours. We specify the bounding box of the rendered map to
be the same as those specified when we created the map object. The default is to save files into the working directory::


    renderer = gs.Renderer(output_dir='./output',
                           output_timestep=gs.hours(6),
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

Save and reload model setup
---------------------------

The ``gnome`` package uses "save files" as a way to save a model setup to use again or to share with another user. The save files are a zip file that contain all the configuration information as JSON files and any needed data files all in one archive. They are usually given the `.gnome` file extension, but they are, in fact, regular zip files.

Save files are used by the WebGNOME application, so that users can save and reload a model setup that they have created via the interactive GUI interface. For the most part, when you are running ``gnome`` via Python scripts, you don't need to use save files, as your script can rebuild the model when it runs. However, there are use cases, particularly if you want to work on the same model via scripting and WebGNOME.

A model can be created from a save file via the
:func:`scripting.load_model()` function:

.. code-block:: python

  import gnome.scripting as gs
  model = gs.load_model("the_savefile.gnome")

You can save out a configured model using the save method:

.. code-block:: python

  model.save("the_savefile.gnome")





