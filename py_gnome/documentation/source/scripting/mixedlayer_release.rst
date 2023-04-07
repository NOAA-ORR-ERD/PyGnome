:orphan:

.. _mixedlayer:

A subsurface (mixed layer) release
==================================


This is a script to look at vertical diffusion mover in the mixed layer
with and without a rise velocity. 


.. code:: python

    import matplotlib.pyplot as plt
    %matplotlib inline
.. code:: python

    from datetime import datetime, timedelta
    
    import numpy as np
    mod
    from gnome.spills.elements import plume
    from gnome.utilities.distributions import UniformDistribution
    
    from gnome.model import Model
    from gnome.map import GnomeMap
    from gnome.spill import surface_point_line_spill
    from gnome.movers import (
                              RiseVelocityMover,
                              RandomMover3D,
                              )
    
    def make_model(rise_vel):
    
        print 'initializing the model'
        start_time = datetime(2015, 8, 1, 0, 0)
        model = Model(start_time=start_time, duration=timedelta(days=5),
                      time_step=5 * 60, uncertain=False)
        
        print 'adding the map'
        model.map = GnomeMap()
        
        print 'adding spill'
        # I use a uniform distribution with the same high/low values to get one size of droplets (50 microns)
        ud = UniformDistribution(low=50e-6,high=50e-6)
        spill = surface_point_line_spill(num_elements=1000,
                                         amount=90,  # default volume_units=m^3
                                         units='m^3',
                                         start_position=(0, 0,
                                                         5),
                                         release_time=start_time,    
                                         element_type=plume(distribution=ud,substance_name='ALASKA NORTH SLOPE (MIDDLE PIPELINE)')
                                         )
        model.spills += spill
    
        # print 'adding a RiseVelocityMover:'
        if rise_vel:
            model.movers += RiseVelocityMover()
    
        # print 'adding a RandomMover3D:'
        model.movers += RandomMover3D(vertical_diffusion_coef_above_ml=50,
                                            vertical_diffusion_coef_below_ml=.11,
                                            mixed_layer_depth=10)
    
        return model
    
.. code:: python

    # With RISE velocity (50 micron droplet)
    model = make_model(rise_vel=True)
    t = np.empty((model.num_time_steps,1000),datetime)
    depths = np.empty((model.num_time_steps,1000),)
    for i,step in enumerate(model):
        t[i,:] = [model.model_time] * 1000
        sc = model.spills.items()[0]
        pos = sc['positions']
        depths[i,:] = -1*pos[:,2]
    
    plt.plot(t[::10],depths[::10],'r.')
    plt.show()

.. parsed-literal::

    initializing the model
    adding the map
    adding spill
    


.. code:: python

    # No rise velocity
    model = make_model(rise_vel=False)
    t = np.empty((model.num_time_steps,1000),datetime)
    depths = np.empty((model.num_time_steps,1000),)
    for i,step in enumerate(model):
        t[i,:] = [model.model_time] * 1000
        sc = model.spills.items()[0]
        pos = sc['positions']
        depths[i,:] = -1*pos[:,2]
    
    plt.plot(t[::10],depths[::10],'r.')
    plt.show()

.. parsed-literal::

    initializing the model
    adding the map
    adding spill
    



