Spills
======
Spills in GNOME contain a release object which specifies the details of the release 
(e.g. where, when, how many elements). They also contain an element_type object which
provides information on the type of substance spilled (e.g. floating oil, subsurface plume etc). 
If element_type is not specified then the default is a conservative floating substance. The 
default for floating substances is to have windage values set in the range 0-4% with a persistence of
15 minutes.

For example, in the scripting :doc:`scripting_intro` we showed how to setup a spill for a conservative substance using
the PointLineRelease class::

    from gnome.model import Model
    from gnome.spill import PointLineRelease, Spill
    from datetime import datetime, timedelta
    start_time = datetime(2015, 1, 1, 0, 0)
    model = Model(start_time=start_time,
                  duration=timedelta(days=3),
                  time_step=60 * 15, #seconds
                  )
    release = PointLineRelease(release_time=start_time,start_position=(-144,48.5,0),num_elements=1000)
    spill = Spill(release)
    model.spills += spill
    
To specify the spill to represent a specific oil from the ADIOS database and specify the amount spilled, we could instantiate the Spill oject like this::
    
    from gnome.spill.elements import ElementType
    element_type=ElementType(substance='ALASKA NORTH SLOPE (MIDDLE PIPELINE)')
    spill = Spill(release,element_type=element_type,amount=5000,units='bbls')
    model.spills += spill
    
But what about this??::

    spill = Spill(release,substance='ALASKA NORTH SLOPE (MIDDLE PIPELINE)',amount=5000,units='bbls')
    
Easier method -- helper functions
---------------------------------

There are some helper functions to simplify setting up typical spills. For example for the case above we could do::

    from gnome.spill import point_line_release_spill
    spill = point_line_release_spill(num_elements=1000,
                                 start_position=(-144,48.5, 0.0),
                                 release_time=start_time,
                                 amount=5000,
                                 substance='ALASKA NORTH SLOPE (MIDDLE PIPELINE)',
                                 units='bbl')
                                 
Special cases
-------------

Subsurface release
~~~~~~~~~~~~~~~~~~
Example subsurface release::

    from gnome.spill import point_line_release_spill
    from gnome.spill.elements import plume
    spill = point_line_release_spill(num_elements=1000,
                                     amount=90,  
                                     units='m^3',
                                     start_position=(-144,48.5, 0.0),
                                     release_time=start_time,
                                     element_type=plume(distribution=ud,substance_name='ALASKA NORTH SLOPE (MIDDLE PIPELINE)')
                                     )
                                     
Conservative particle, specified windage range
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Here the windage parameters are set to be in the range 0-1% and have infinite persistence 
(once the windage value is assigned to an LE is keeps that value for all time)::

    spill = gnome.spill.point_line_release_spill(num_elements=5000,
            start_position=(-144,48.5, 0.0),
            release_time=start_time,
            element_type=floating(windage_range=(0,1),windage_persist=-1))