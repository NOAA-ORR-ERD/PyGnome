Movers
======

Processes that change the position of the LEs are termed "movers" in GNOME. These can include advection of the LEs due to winds and currents, 
diffusive movement of LEs due to unresolved turbulent flow fields, and prescribed behavior of the LEs (e.g. rise velocity of oil droplets 
or larval swimming.)

Some examples and common use cases are shown here. For complete documentation see :mod:`gnome.movers`

Wind movers
-----------

Wind movers are tied to a wind object in the Environment class introduced in the :doc:`weatherers` section.

For example to create a wind mover based on manually entered time series::

    from gnome.model import Model
    from gnome.environment import Wind
    from gnome.movers import WindMover
    from gnome.basic_types import datetime_value_2d
    import numpy as np
    from datetime import datetime, timedelta
    start_time = datetime(2004, 12, 31, 13, 0)
    series = np.zeros((2, ), dtype=datetime_value_2d)
    series[0] = (start_time, (5, 180))
    series[1] = (start_time + timedelta(hours=18), (5, 180))

    model = Model()
    wind = Wind(timeseries=series,units='m/s')
    w_mover = WindMover(wind)
    model.movers += w_mover


Current movers
--------------

Random movers
-------------

Rise velocity mover
-------------------