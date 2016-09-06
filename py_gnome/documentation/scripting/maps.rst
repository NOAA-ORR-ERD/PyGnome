Maps
====

Maps can be created manually (e.g. in the case of all water), generated from a model grid, or loaded from a file. 
The map file format supported by GNOME is the BNA shoreline format (see the GNOME file formats document). 
Support for shapefiles is currently being implemented.

If a BNA map is loaded this will be the shoreline used for oil beaching algorithms. If gridded movers are also
added to the model, it may be desirable to have the shoreline generated from the currents grid so that it matches
(both in exact position and in resolution). See the section below on Generating Map from Currents Grid.

Some of the more common use cases and map options are shown below. 
For complete documentation see :mod:`gnome.map`

Manual Creation
---------------
Create an all water map -- if map_bounds is not specified, a spill can be added anywhere in the "water world"::

    from gnome.map import GnomeMap
    mymap = GnomeMap()

*Options:*

map_bounds
    A polygon bounding the map, e.g. map_bounds = ((-145,48), (-145,49), (-143,49), (-143,48))

Load Shoreline from File
------------------------
Load a BNA map file::

    from gnome.map import MapFromBNA
    mymap = MapFromBNA(mapfile.bna, refloat_halflife=1) 

*Options:*

refloat_halflife
    Particles that beach on the shorelines are randomly refloated according to the specified half-life 
    (specified in hours). This is a global shoreline parameter, i.e. there is only one shoreline type.
    If no refloating is desired set this value to -1.

Create Map from Currents Grid
-----------------------------

This feature is under development in pygnome.

Add Map to Model
----------------
This can be done a few ways. For example, when setting up the model::

    model = Model(time_step=3600, start_time=datetime(2015, 8, 1, 0, 0),
    duration=timedelta(days=1),
    map=mymap)
                  
Or added to the model after it has been set up.::

    model.map = mymap 
                 