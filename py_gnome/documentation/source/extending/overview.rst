Extending / Customizing PyGNOME
===============================

PyGNOME is a system build of standardized components. The Model's job is to coordinate each of the components into a fully working model, but the details are all determined at run time according to what components the model has been configured to use.

As a result you can write a new component and plug it in to the rest of the model without having to alter any of the model or other code.

The multiple PyGNOME APIs
-------------------------

The gnome package is designed to work when driven by scripts, as well as the engine behind the WebGNOME system. This means that there are a couple of related APIs for each GnomeObject:

The scripting API
    This is how an object is created / manipulated in python scripts. When making new objects, you can use whatever you like, but it's probably a good idea to make the API similar to other existing objects.

The model API
    These are the methods and attributes the object needs to have in order to work with the ``gnome.Model`` system. For the most part, the core methods will be defined in a Base Class (e.g. the ``gnome.movers.mover.Mover`` class), so if you subclass from that, you only need to implement the ones you need.

The Web / Save file API
    WebGNOME use a JSON API to configure the PyGNOME system. A very similar JSON format is used to save model configurations in "save files", which are zip files with the model configuration and data required to reproduce a model configuration. You can develop and test your new component without using this system, but if you want your component to be usable with WebGNOME or save files, then This API must be defined. This is done by defining a ``Schema`` that specifies what attributes need to be saved and can be updated for a given object. See: :ref:`serialization_overview` for the details.


Movers
------

A mover is component that "moves" the elements in the model. PyGNOME supports having multiple movers that are all combined via linear superposition. There are some details, but the key method a Mover needs to provide is ``get_move()``:

.. code-block:: python

    def get_move(self, sc, time_step, model_time_datetime):
        """
        Compute the move in (long, lat, z) space. It returns the delta move
        for each element of the spill as a numpy array of size
        (number_elements X 3) and dtype = gnome.basic_types.world_point_type

        Base class returns an array of numpy.nan for delta to indicate the
        get_move is not implemented yet.

        Each class derived from Mover object must implement it's own get_move

        :param sc: an instance of gnome.spill_container.SpillContainer class
        :param time_step: time step in seconds
        :param model_time_datetime: current model time as datetime object

        All movers must implement get_move().
        """
        positions = sc['positions']

        delta = np.zeros_like(positions)
        delta[:] = np.nan

        return delta

Note that `get_move()` doesn't alter the positions of the elements. rather it returns a "delta" -- of the amount that of movement in (longitude, latitude, vertical meters) units) for the provided time step. This is so that the model can apply the "delta" from each mover and the result won't change with the order in which they are applied. Essentially, the delta values from all the movers are added together, and then added to the positions array.

``sc`` is a ``SpillContainer``, it contains all the arrays of data associated with the elements. For the most part, for a mover, the "positions" array is the important one, but if movement is affected by other properties of the elements, that data will be stored in the spill_container. The ``SpillContainer`` is a complex object that manges all the data associated with the elements. But it presents a "Mapping" interface (like a Pyton dict) for easy access to the data associated with the elements -- for example: ``sc['postions']`` will return a 3xN numpy array with the positions of the elements. ``sc['mass']`` wil return an array with the mass of each element, etc. Which arrays are assocaed with the elements depends on what components have been added to the model.

the ``get_move()`` method will also have access to the time step length (in seconds), and the model time at the beginning of this time step.



Writing a custom mover
......................

[To be filled in]

Here's how to do this.

And here's some example code doing this::

    from gnome.model import Model
    model = Model()

That doesn't do much. You can find out more in :mod:`gnome.movers`


Weatherers
----------

Weatherers are objects that alter the elements. The name comes from the oil weathering code shipped with PyGNOME, but they can be used for any process that changes the properties of the elements.

The Base class for weatherers is :class:`gnome.weatherers.core.Weatherer`

The core method each weatherer needs is:

.. code-block: python

    def weather_elements(self, sc, time_step, model_time):

weatherers update the data arrays in the spill container, ``sc``.

NOTE: this is different than movers -- as each weatherer updates the actual data associated with the elements, the result will be different depending on what order they are run. This order is controlled by ``sort_order`` in ``gnome/weatherers/__init__.py``


Making your own weatherer
.........................

And here's how to do this...


Map
---

The Model has to have a Map at all times. After moving and weathering the elements, the map is responsible for the interaction between the elements and the shoreline, as well as the ocean bottom and water surface. (bottom and water surface are only relevant for 3-d simulations).

It will determine whether elements have impacted the shoreline, or gone off the map, and can also refloat any elements that were previously beached.

The base map object: :class gnome.GnomeMap: represents a "water world" -- no land anywhere, and unlimited map bounds. But it also provides a base class with the full API. To create another map object, you derive from GnomeMap, an override the methods that you want to implement in a different way. In addition, it should have a few attributes used by the model:


Key Attributes of Maps
......................

``map_bounds``: The polygon bounding the map if any elements are outside the map bounds, they are removed from the simulation.

``spillable_area``: The PolygonSet bounding the spillable_area. Either a PolygonSet object or a list of lists from which a polygon set can be created. Each element in the list is a list of points defining a polygon.

``land_polys``: The PolygonSet holding the land polygons. These are only used for display. Either a PolygonSet object or a list of lists from which a polygon set can be created. Each element in the list is a list of points defining a polygon.

Each of these defaults to the whole world with no land.

Key methods
...........

The key methods to override include:

``__init__``: Whatever you need to initialize your map object

.. code-block: python

    beach_elements(spill):

        Determines which elements were or weren't beached or moved off_map.

        It is passed a "SpillContainer", which is essentially a dictionary that
        holds the data associated with the elements.

        Called by the model in the main time loop, after all movers have acted.

        spill['status_code'] is changed to oil_status.off_maps if off the map.

        :param spill: current SpillContainer
        :type spill:  :class:`gnome.spill_container.SpillContainer`

        subclasses that override this probably want to make sure that:

        self.resurface_airborne_elements(spill)
        self._set_off_map_status(spill)

        are called. Unless those functions are overloaded

.. code-block:: python

    refloat_elements(self, spill_container, time_step):
        """
        This method performs the re-float logic -- changing the element
        status flag, and moving the element to the last known water position

        :param spill_container: current SpillContainer
        :type spill_container:  :class:`gnome.spill_container.SpillContainer`
        """
