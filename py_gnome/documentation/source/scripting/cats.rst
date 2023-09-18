.. _tutorial-2:

CATS currents
==============

How to use GNOME with currents from the NOAA's CATS model

NOAA's CATS model is a hydrodynamic model used to model current patterns in most of the Location Files.
To create a script to run a location file start with one of the testing scripts in the scripting module.
The testing scripts include a few examples of Location Files - Boston, Delaware Bay, Long Island, and Mississippi.
These have CATS patterns tied to different types of tides or just scaled steady state patterns.
The files can be extracted from a save file exported from WebGnome. These files are also part of the WebGnomeAPI project.
For example, Long Island Sound has a single CATS pattern that is tied to a tide. If the reference point you want to scale to
is different from the one in the tide file you can set the reference point and scaling after creating the CatsMover.

.. code-block:: python

    import gnome.scripting as gs

    c_mover = gs.CatsMover('LI_tidesWAC.CUR', tide=gs.Tide('CLISShio.txt'))

    model.movers += c_mover
    model.environment += c_mover.tide

The Mississippi River Location File, has a simple scaled CATS pattern where the scale value is based on the stage height or surface current of the river at New Orleans.

.. code-block:: python

    import gnome.scripting as gs

    c_mover = gs.CatsMover('LMiss.CUR')

    # need to scale based on river stage height
    c_mover.scale = True
    c_mover.scale_refpoint = (-89.699944, 29.494558)

    # based on stage height 10ft (range is 0-18)
    c_mover.scale_value = 1.027154

    model.movers += c_mover

The Delaware Bay Location File includes a component mover, which is composed of wind driven current patterns.
For example, from the Delaware Bay Location File:

.. code-block:: python

    import gnome.scripting as gs
    import numpy as np
    from datetime import datetime
    from gnome.basic_types import datetime_value_2d

    series = np.zeros((2, ), dtype=datetime_value_2d)
    series[0] = (start_time, (5, 270))
    series[1] = (start_time + timedelta(hours=25), (5, 270))

    wind = gs.PointWind(timeseries=series, units='m/s')

    comp_mover = ComponentMover('NW30ktwinds.cur', 'SW30ktwinds.cur', wind)

    comp_mover.scale_refpoint = (-75.263166, 39.1428333)

    comp_mover.pat1_angle = 315
    comp_mover.pat1_speed = 30
    comp_mover.pat1_speed_units = 1
    comp_mover.pat1_scale_to_value = .502035

    comp_mover.pat2_angle = 225
    comp_mover.pat2_speed = 30
    comp_mover.pat2_speed_units = 1
    comp_mover.pat2_scale_to_value = .021869

    model.movers += comp_mover

 These examples can be run as scripts with the addition of map, spill, and outputter.
