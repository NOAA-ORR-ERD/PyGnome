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
* spills
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
--------------------------------------

.. automodule:: gnome.map
    :members:

``gnome.spill`` -- classes in the spill module
-----------------------------------------------

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
----------------------------------------------------------------------

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
----------------------------------------------------
.. automodule:: gnome.environment
.. autoclass:: Tide
    :members:
    :inherited-members:
.. autoclass:: Wind
    :members:
    :inherited-members:

``gnome.environment.environment_objects`` -- PyGnome implemented environment objects
------------------------------------------------------------------------------------

.. .. automodule:: gnome.environment.environment_objects

.. autoclass:: GridCurrent
    :members: from_netCDF, at
    :show-inheritance:
    :inherited-members: from_netCDF, at
    :undoc-members: default_names

.. autoclass:: GridWind
    :members:
    :show-inheritance:
    :undoc-members: default_names

.. autoclass:: IceConcentration
    :members:
    :show-inheritance:
    :undoc-members: default_names

.. autoclass:: GridTemperature
    :members:
    :show-inheritance:
    :undoc-members: default_names

.. autoclass:: IceAwareCurrent
    :members:
    :show-inheritance:
    :undoc-members: default_names

.. autoclass:: IceAwareWind
    :members:
    :show-inheritance:
    :undoc-members: default_names

.. autoclass:: TemperatureTS
    :members:
    :show-inheritance:



``gnome.environment.gridded_objects_base`` -- PyGnome wrappers for gridded objects
----------------------------------------------------------------------------------

.. .. autoclass:: Time
..     :members:
.. automodule:: gnome.environment.gridded_objects_base
.. autoclass:: Variable
    :members: at, from_netCDF
    :show-inheritance:
.. autoclass:: VectorVariable
    :members: at, from_netCDF
    :show-inheritance:
.. automodule:: gnome.environment.ts_property
.. autoclass:: TimeSeriesProp
    :members:
    :show-inheritance:
    :inherited-members:
.. .. autoclass:: TSVectorProp
..    :members:
..    :show-inheritance:
..    :inherited-members:

``gnome.outputter`` -- PyGnome outputters module
------------------------------------------------

.. automodule:: gnome.outputters
.. autoclass:: Outputter
    :members:
    :show-inheritance:
.. autoclass:: Renderer
    :members:
    :show-inheritance:
.. autoclass:: NetCDFOutput
    :members:
    :show-inheritance:
.. autoclass:: KMZOutput
    :members:
    :show-inheritance:
.. autoclass:: TrajectoryGeoJsonOutput
    :members:
    :show-inheritance:
.. autoclass:: IceGeoJsonOutput
    :members:
    :show-inheritance:

``gnome.utilities`` -- PyGnome utilities module
---------------------------------------------------

.. automodule:: gnome.utilities.distributions
    :members:
    :show-inheritance:
.. This gave some errors -- not sure why
.. .. automodule:: gnome.utilities.serializable
..    :members:
..    :show-inheritance:
.. automodule:: gnome.utilities.orderedcollection
    :members:
    :show-inheritance:
.. automodule:: gnome.utilities.map_canvas
    :members:
    :show-inheritance:
.. automodule:: gnome.utilities.projections
    :members:
    :show-inheritance:
.. automodule:: gnome.utilities.inf_datetime
    :members:
    :show-inheritance:
.. automodule:: gnome.utilities.cache
    :members:
    :show-inheritance:



``gnome.persist`` -- PyGnome persistence classes
------------------------------------------------
.. automodule:: gnome.persist.base_schema
    :members:

.. automodule:: gnome.persist.extend_colander
    :members:

.. automodule:: gnome.persist.validators
    :members:

Persistance Glossary
====================

These terms mostly come from the ``Colander`` library

.. NOTE: copy and pasted from the Colander docs -- we were getting errors with glossary entries in docstrings that got pulled from the superclass methods.

.. glossary::
   :sorted:

   cstruct
     A data structure generated by the
     :meth:`colander.SchemaNode.serialize` method, capable of being
     consumed by the :meth:`colander.SchemaNode.deserialize` method.

   appstruct
     A raw application data structure (a structure of complex Python
     objects), passed to the :meth:`colander.SchemaNode.serialize`
     method for serialization.  The
     :meth:`colander.SchemaNode.deserialize` method accepts a
     :term:`cstruct` and returns an appstruct.

   schema
     A nested collection of :term:`schema node` objects representing
     an arrangement of data.

   schema node
     A schema node is an object which can serialize an
     :term:`appstruct` to a :term:`cstruct` and deserialize a
     :term:`appstruct` from a :term:`cstruct` an (object derived from
     :class:`colander.SchemaNode` or one of the colander Schema
     classes).

   type
     An object representing a particular type of data (mapping,
     boolean, string, etc) capable of serializing an :term:`appstruct`
     and of deserializing a :term:`cstruct`.  Colander has various
     built-in types (:class:`colander.String`,
     :class:`colander.Mapping`, etc) and may be extended with
     additional types (see Colander docs).

   validator
     A Colander validator callable.  Accepts a ``node`` object and a
     ``value`` and either raises an :exc:`colander.Invalid` exception
     or returns ``None``.  Used as the ``validator=`` argument to a
     schema node, ensuring that the input meets the requirements of
     the schema.  Built-in validators exist in Colander
     (e.g. :class:`colander.OneOf`, :class:`colander.Range`, etc), and
     new validators can be defined to extend Colander (see
     Colander docs).

