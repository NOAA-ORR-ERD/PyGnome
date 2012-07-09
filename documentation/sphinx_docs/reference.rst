.. _reference:

``pyGNOME`` Reference
===========================

pyGNOMEis a wrapper around a set of C++ libraries. THe C++ code is designed to be used by itself, or from the Pyton, though using it Python is easier. The API is perhaps a bit klkunky -- the C++ code was all orginally written as part of monolithic GUI program -- we have separated out the GUI parts, and cleaned up the API so that the pieces can be used individually.

Basic Structure
------------------

There a handlful of core base classes you need to use pyGNOME for anything useful:

a model:
   This is the main class that keeps track of all teh pieces, runs the loop through time, etc. The code comes with a full-featured version -- you may want a simmpler one if you aren't do a full-on oil spill model.

movers:
   These are classes that represent anything that moves a particle. Or, on fact, alters a particle in any way (the name mover is abit hiostorical -- they used to only moved particles...) Examples include surface winds, currents (form a variety of sources), weathering processes, etc. -- this is where the real work is done. Each mover's action is essentially linear superposed with the others. i.e at each tiem step, the model loops through all the movers, and passes the spill objects to be acted on.
   At the C++ level, each mover has an `get_move` method that takes the curent time, model time step, and pointers to the arrays of particle properties it needs. At the Python levevel, the get_move method takes a spill object, and the required arrays are extracted form teh spill object -- this lets us pass only that data to a mover that it really needs, and lets us use Python for the dynamic parts -- making sure that the data needed exists.

spills:
   A spill class is a class that holds a set of particles and various information about them. each of the particle properties are stored as numpy arrays (in a dict) -- so that for a given model setup , the spill only needs to hae the properties required, and the properties used by a given mover (and only those) can be passed in as a C pointer to the mover code. At the very least, each spill has a set of particle position arrays.
   There may be multiple spills in a model set-up, but for efficiency'y's sake, each spill usually is a set of 1000 or so particles that share various properties.
  A spill class has a `release_particles` method that is called at each tiem step, so that the number of particles can increase as time goes on, etc.

a map:
   A map keeps track of where land and water are. Teh siimplest map is all teh earth with no land. It has methods to ask if a locatino ison land, if a locatioin is "spillable", etc. The most comonnly used map for surface oil spills is intialized with a `*.bna` file describing parlygons of land -- this is rasterized into a land-eater bitmap. During teh run, the model calls the `'beach_LEs` method, which determines which particles have hit land in the last tioiem step, and sets htose particles to "beached".


Class Reference:
---------------------

.. automodule:: gnome
   :members:

``gnome.model`` -- the core main model class
--------------------------------------------------
.. automodule:: gnome.model
   :members:
      
``gnome.map`` -- the map classes
---------------------------------------------------
.. automodule:: gnome.map
   :members:







