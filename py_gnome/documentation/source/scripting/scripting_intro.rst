.. _scripting_overview:

Overview
========

PyGNOME is a complex package, however, it is fairly straightforward to write scripts to run the model for a variety of simulations.

Simulations with the ``gnome`` package are performed via a ``Model`` object. A model has model-level attributes such as the start time, run duration, and time step. It can be configured with a variety of  ``Mover``s, ``Weatherers``, and ``Spill``s to drive the model, and ``Outputter``s can output the model results in a number of formats.


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

    Internally, ``GNOME`` uses the python standard library ``datetime`` and ``timedelta`` functions.
    In most cases, you can pass objects of these types into ``GNOME``.
    But for scripting convenience, most places that take a datetime object will also accept a ISO 8601 string, such as: "2015-01-01T12:15:00"

The ``gnome.scripting`` module also provides a number of shortcuts for creating ``timedelta`` objects: ``seconds, minutes, hours, days``. You can use them like so::

    gs.hours(3)  # for 3 hours
    gs.days(2)  # for two days

Examples in this section will use the scripting module. In the following more detailed sections about specific object types, we may sometimes also show the full import path for clarity.

Simplest Example
----------------

For a first introduction to scripting with ``gnome``,
the example below will demonstrate how to set up and run the very simplest model possible.
This example does not load any external data, but creates a simple map and movers manually.
The "spill" is a conservative substance, i.e. a "passive tracer"

This example is in the PyGNOME source under "scripts/example_scripts", or can be downloaded here:
.. this download link is getting rendered oddly -- and why do I need so many ../?
:download:`simple_script.py <../../../../../scripts/example_scripts/simple_script.py`

Initialize the Model
--------------------
The model is initialized to begin on New Years Day 2015 and run for 3 days with a model time step of 15 minutes::

    import gnome.scripting as gs
    start_time = "2015-01-01"
    model = gs.Model(start_time=start_time,
                     duration=gs.days(3),
                     time_step=gs.minutes(15)
                     )


Create and Add a Map
--------------------
Create a very simple map which is all water with a rectangle defined by four longitude/latitude points to specify boundary of the model::

    model.map = gs.GnomeMap(map_bounds=((-145, 48), (-145, 49),
                                        (-143, 49), (-143, 48))
                                        )

Create and Add Movers
---------------------
THe model needs one or more "Movers" to move the elements. In this case, a steady uniform current and random walk diffusion are demonstrated.

The `SimpleMover` class is used to specify a 0.2 m/s eastward current.

The `RandomMover` class simulates spreading due to turbulent motion via a random walk algorithm:

.. code-block:: python

    velocity = (.2, 0, 0)  # (u, v, w) in m/s
    uniform_vel_mover = gs.SimpleMover(velocity)
    #  random walk diffusion -- diffusion_coef in units of cm^2/s
    random_mover = gs.RandomMover(diffusion_coef=2e4)

    # the movers are added to the model
    model.movers += uniform_vel_mover
    model.movers += random_mover


Create and Add a Spill
----------------------

Spills in ``gnome`` specify what, when, where, and how many elements are released into the model. The properties of the substance spilled (e.g. oil chemistry) are provided by a ``Substance`` Object. PYGNOME currently has two Substances available: ``NonWeatheringSubstance`` representing passive drifters, and ``GnomeOil``, representing petroleum products with all the properties required for the oil weathering algorithms supplied with GNOME.

There are a number of "helper" functions and classes that can initialize various types of spills (for example, at a point or over a spatial area, at the surface or subsurface). See: :ref:`scripting_spills` for more details.
 
A common spill type is created by the `surface_point_line_spill`. To set up a instanatious release of a conservative substance at a point, it can be called with most of the defaults::


    spill = gs.surface_point_line_spill(release_time=start_time,
                                        start_position=(-144, 48.5),
                                        num_elements=500)
    model.spills += spill

* The release time is set to the start_time previous defined to start the model.
* The release location (start_position) is set to a (longitude, latitude) position.
* The number of Lagrangian elements (particles) can be defined (defaults to 1000)

Create and Add an ``Outputter``
-------------------------------

Outputters save the model results in a variety of formats.
Options include PNG images and saving the element information into netCDF files, shapefiles, or KML for further visualization and analysis. See :ref:`scripting_outputters`

In this example, the ``Renderer`` class is used to save to an animated gif every 3 hours::

    renderer = gs.Renderer(output_dir='./output/',
                           output_timestep=gs.hours(2),
                           # bounding box for the output images
                           viewport=((-145, 48), (-145, 49),
                                     (-143, 49), (-143, 48)),
                           formats=['gif']
                           )

    model.outputters += renderer

* The time step for output is set to 2 hours.

* The bounding box (viewport) of the rendered map is set to be the same as those specified for the map object.

* ``Renderer`` supports 'bmp', 'jpg', 'jpeg', 'png' and 'gif' -- 'gif' will save out a single animated GIF file - the rest will output one image per output timestep.


Run the Model
-------------

Once the model is all set up, the simulation can be run.

To run the model for the entire duration::

    model.full_run()

Results will be written to files based on the outputters added to the model -- in this case, an animated GIF named ``anim.gif``.

View the results
----------------

The renderer added to the model generates an animated GIF with a frame every 8 hours as specified in its creation.

It will have been saved in ``output`` dir relative to the directory that the script was executed from, as specified in the ``Renderer`` creation.
The animation should show a cloud of elements moving east and spreading.
