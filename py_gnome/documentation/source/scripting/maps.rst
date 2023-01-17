.. include:: ../links.rst

Maps
====

Maps can be created manually (e.g. in the case of all water), generated from a model grid, or loaded from a file. 
The map file format supported by PyGNOME is the BNA shoreline format (see the |file_formats_doc|). 
Support for shapefiles is currently being implemented.

If a BNA map is loaded this will be the shoreline used for oil beaching algorithms. If gridded movers are also
added to the model, it may be desirable to have the shoreline generated from the currents grid so that it matches
(both in exact position and in resolution). 

Some of the more common use cases and map options are shown below. We show the full path imports 
for clarity but use the scripting module discussed in the previous section to create objects in 
our example scripts.

For complete documentation on Maps see :mod:`gnome.map`

Manual Creation :class:`gnome.map.GnomeMap`
-------------------------------------------
Create an all water map -- if map_bounds is not specified, a spill can be added anywhere in the "water world"::

    import gnome.scripting as gs
    mymap = gs.GnomeMap()

*Useful options:*

map_bounds
    A polygon bounding the map, e.g. map_bounds = ((-145, 48), (-145, 49), (-143, 49), (-143, 48))

Load Shoreline from File :class:`gnome.map.MapFromBNA`
------------------------------------------------------
Load a BNA map file::

    mymap = gs.MapFromBNA('mapfile.bna', refloat_halflife=1) 

*Useful Options:*

refloat_halflife
    Particles that beach on the shorelines are randomly refloated according to the specified half-life 
    (specified in hours). This is a global shoreline parameter, i.e. there is only one shoreline type.
    If no refloating is desired set this value to -1.

Create Map from Currents Grid
-----------------------------

This feature is under development.

Add Map to Model
----------------
This can be done a few ways. For example, when setting up the model::

    model = gs.Model(time_step=3600, start_time='8/1/2015',
                     duration=gs.days(1),
                     map=mymap)
                  
Or added to the model after it has been set up.::

    model.map = mymap 
                 