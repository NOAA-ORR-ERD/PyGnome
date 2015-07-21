.. _reference:

PyGnome Reference
===========================

PyGnomeis a wrapper around a set of C++ libraries. THe C++ code is designed to
be used by itself, or from Python, though using it in Python is easier. The API
is perhaps a bit klunky -- the C++ code was all orginally written as part of a
monolithic GUI program -- we have separated the GUI parts, and cleaned up the
API so that the pieces can be used individually.

Basic Structure
-------------------

There are a handful of core base classes you need to use PyGnome for anything
useful:

* `a model <reference.html#model>`_
* movers:
   These are classes that represent anything that moves a particle.  Or in fact,
   alters a particle in any way (the name mover is a bit historical -- they
   used to only move particles...).  Examples include surface winds, currents
   (from a variety of sources), weathering processes, etc. -- this is where the
   real work is done.  Each mover's action is essentially linear superposed
   with the others. i.e at each time step, the model loops through all the
   movers, and passes the spill objects to be acted on.
   At the C++ level, each mover has a `get_move` method that takes the current
   time, model time step, and pointers to the arrays of particle properties it
   needs.  At the Python level, the get_move method takes a spill object, and
   the required arrays are extracted from the spill object -- this lets us pass
   to a mover only the data that it really needs, and lets us use Python for
   the dynamic parts -- making sure that the data needed exists.
* spills:
   A spill class is a class that holds a set of particles and various
   information about them. Each of the particle properties are stored as numpy
   arrays (in a dict) -- so that for a given model setup, the spill only needs
   to have the properties required, and the properties used by a given mover
   (and only those) can be passed in as a C pointer to the mover code.  At the
   very least, each spill has a set of particle position arrays.

   There may be multiple spills in a model set-up, but for efficiency's sake,
   each spill usually is a set of 1000 or so particles that share various
   properties.
   
   Each spill is a composition of the type of substance spilled (ElementType)
   and how elements are released (instantaneous, continuous, etc).
* a map:
   A map keeps track of where land and water are.  The simplest map is all the
   earth with no land.  It has methods to ask if a location is on land, if a
   location is "spillable", etc.  The most commonly used map for surface oil
   spills is intialized with a `*.bna` file describing polygons of land -- this
   is rasterized into a land-eater bitmap.  During the run, the model calls the
   `'beach_LEs` method, which determines which particles have hit land in the
   last time step, and sets those particles to "beached".

Class Reference:
-------------------

.. include:: model.rst
------------------------

``gnome.model`` -- the PyGnome model class
---------------------------------------------------
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