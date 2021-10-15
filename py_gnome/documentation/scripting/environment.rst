.. include:: ../links.rst
Environment Objects
===================

As transport and weathering of particles in PyGNOME depend on a variety of environmental conditions 
(e.g. wind, waves, and water properties),
initialization of various environment objects is needed before these processes can be added to
the model. In some cases, these objects may be automatically created. For example,when creating a gridded 
current mover from a NetCDF file (next section). Sometimes, it is necessary or desirable to manually
create these objects. For example, if weatherering and transport processes may be dependent on the 
same environmental information (winds) or if you want to enter data manually.

Environment objects provide an interface that can be queried in time and space. They
can represent a spatially constant time series or gridded, time dependent data. 

Examples of conditions that environment objects can represent include: temperature, water velocity, wind speed & direction time series.

For detailed documentation of the API and implemented objects see :mod:`gnome.environment.environment_objects`

Interacting with The Environment Class
--------------------------------------

Here's a detailed example to create a simple Wind object (for a constant in time wind). We'll take advantage of the gnome scriping 
module to avoid having to manually import the necessary classes and functions::

    import gnome.scripting as gs
    import numpy as np
    from datetime import datetime
    from gnome.basic_types import datetime_value_2d
    model = gs.Model(start_time="2015-01-01",
             duration=gs.days(3),
             time_step=gs.minutes(15)
             )
    series = np.zeros((1, ),dtype=datetime_value_2d) #Make a wind time series (one value for wind constant in time)
    series[0] = (datetime(2015,1,1,0), (10, 0))
    wind = gs.Wind(timeseries=series,units='knots')

This is still rather complicated. Much more simply, we can use the helper function for creating a constant wind::

    wind = gs.constant_wind(10,0,'knots')
    
Alternatively, if we had a properly formatted file (|file_formats_doc|) with a timeseries of wind data at a single point, we could use that to create a Wind Object using the Wind Class that is imported into the scripting module for convenience. An example file that was used for this ::

    wind = gs.Wind(filename='wind_file.txt')

.. admonition:: Environment Objects

    An important note, is that environment objects alone do not have any effect on the model simulation. Once they are created, they can be explicitly passed to weatherers and movers. However, if a weatherer is added to the model without explicity specifying the required environment objects, then the first object of the correct type in the environment collection will be used for that weathering process. 
    For example, if multiple wind time series are created and added to model.environment then the first one added will be used
    for weathering processes unless explicitly specified.

Example of adding a manually adding a wind object to the model enviornment::

    model = gs.Model()
    model.environment += wind
    
More examples of the interaction of environment objects with movers and weatherers will be given in the next section.





