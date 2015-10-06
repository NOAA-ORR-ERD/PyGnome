PyGnome Class Reference
=======================
There are a handful of core base classes in PyGnome.

``gnome.model`` -- the PyGnome model class
------------------------------------------
This is the main class that contains objects used to model trajectory and
weathering processes. It runs the loop through time, etc.
The code comes with a full-featured version -- you may want a simpler one if
you aren't doing a full-on oil spill model. The model contains:

* map
* collection of environment objects
* collection of movers
* collection of weatherers
* a spills
* its own attributes

In pseudocode, the model loop is defined below. In the first step, it sets up the
model run and in subsequent steps the model moves and weathers elements. 

.. code-block:: python

    start_at_step_num = -1
    
    if step_num == -1
        setup_model_run()
    
    else:
        setup_time_step()
        move()
        weather()
        step_is_done()
    
    step_num += 1
    
    for sc in self.spills.items():
        num = sc.release_elements()     # initialize mover data arrays
        
        if num > 0:
            for w in weatherers:
                w.initialize_data()     # initialize weatherer data arrays
    
    o = write_output()
    
    return o
    
.. automodule:: gnome.model
.. autoclass:: Model
   :members:
   :undoc-members:
   :show-inheritance:
      
``gnome.map`` -- the PyGnome map class
---------------------------------------------------
.. automodule:: gnome.map
   :members:

``gnome.spill`` -- classes in the spill module
---------------------------------------------------
.. automodule:: gnome.spill
.. autoclass:: Spill
   :members:
   :inherited-members:
.. autoclass:: Release
   :members:
   :inherited-members:
.. autoclass:: PointLineRelease
   :members:
   :inherited-members:
.. autoclass:: SpatialRelease
   :members:
   :inherited-members:
.. autoclass:: VerticalPlumeRelease
   :members:
   :inherited-members:

``gnome.spill.elements`` -- classes in the elements module
--------------------------------------------------------------
.. automodule:: gnome.spill.elements.element_type
.. autoclass:: ElementType
   :members:
   :inherited-members:
.. autofunction:: floating
.. autofunction:: plume

``gnome.movers`` -- PyGnome mover classes
---------------------------------------------------
.. automodule:: gnome.movers
.. autoclass:: Process
   :members:
.. autoclass:: Mover
   :members:
.. autoclass:: CyMover
   :members:
.. autoclass:: RandomMover
   :members:
   :inherited-members:
.. autoclass:: GridCurrentMover
    :members:
    :inherited-members:
.. autoclass:: WindMover
   :members:
   :show-inheritance:
.. autoclass:: GridWindMover
    :members:
    :inherited-members:

``gnome.weatherers`` -- PyGnome/Adios weathering/mass removal classes
-------------------------------------------------------------------------
.. automodule:: gnome.weatherers
.. autoclass:: Weatherer
   :members:
.. autoclass:: Evaporation
   :members:
.. autoclass:: Emulsification
   :members:
   :inherited-members:
.. autoclass:: NaturalDispersion
    :members:
    :inherited-members:
.. autoclass:: Skimmer
   :members:
   :show-inheritance:
.. autoclass:: Burn
    :members:
    :inherited-members:


``gnome.environment`` -- PyGnome environment classes
-------------------------------------------------------
.. automodule:: gnome.environment
.. autoclass:: Tide
    :members:
    :inherited-members:
.. autoclass:: Wind
    :members:
    :inherited-members:

``gnome.outputter`` -- PyGnome outputters module
---------------------------------------------------
.. automodule:: gnome.outputters
.. :autoclass:: Outputter
    :members:
    :show-inheritance:
.. :autoclass:: Renderer
    :members:
    :show-inheritance:
.. :autoclass:: NetCDFOutput
    :members:
    :show-inheritance:
.. autoclass:: TrajectoryGeoJsonOutput
    :members:
    :show-inheritance:
.. autoclass:: CurrentGeoJsonOutput
    :members:
    :show-inheritance:
.. autoclass:: IceGeoJsonOutput
    :members:
    :show-inheritance:

``gnome.utilities`` -- PyGnome utilities module
---------------------------------------------------

.. automodule:: gnome.utilities.serializable
    :members:
    :show-inheritance:

.. automodule:: gnome.utilities.orderedcollection
    :members:
    :show-inheritance:


``gnome.persist`` -- PyGnome persistance classes
---------------------------------------------------
.. automodule:: gnome.persist.base_schema
   :members:
.. automodule:: gnome.persist.extend_colander
   :members:
.. automodule:: gnome.persist.validators
   :members: