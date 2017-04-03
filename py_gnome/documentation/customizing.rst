Customizing pygnome
===================

The pygnome Mover and Weatherer APIs are defined so that you can write your own movers and/or weatherers.

Writing a custom mover
----------------------
Here's how to do this.

And here's some example code doing this::

    from gnome.model import Model
    model = Model()

That doesn't do much. You can find out more in :mod:`gnome.movers`


Writing a custom weatherer
--------------------------

And here's how to do this...

Writing a custom Map
---------------------

The Model has to have a Map at all times. After moving and weathering the elements, the map determines whether elements have impacted the shoreline, or gone off the map, and also refloats any elements that should be refloated.

The base map object: :class gnome.GnomeMap: represents a "water world" -- no land anywhere, and unlimited map bounds. But it also provides a base class with the full API. To create another map object, you derive from GnomeMap, an override the methods that you want to implement in a different way. In addition, it should have a few attributes used by the model:

Key Attributes
..............

``map_bounds``: The polygon bounding the map if any elements are outside the map bounds, they are removed from the simulation.

``spillable_area``: The PolygonSet bounding the spillable_area. Either a PolygonSet object or a list of lists from which a polygon set can be created. Each element in the list is a list of points defining a polygon.

``land_polys``: The PolygonSet holding the land polygons. These are only used for display. Either a PolygonSet object or a list of lists from which a polygon set can be created. Each element in the list is a list of points defining a polygon.

Each of these defaults to the whole world with no land.

Key methods
...........

The key methods to override include:

``__init__``: Whatever you need to initialize your map object

::

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
