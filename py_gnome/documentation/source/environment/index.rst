.. marked as orphan so we don't get warned about it not being a toctree

:orphan:

Using your data
===============

.. toctree::
   :maxdepth: 2

   env_obj_intro
   glossary
   environment_objects
   env_obj_examples
..   examples.ipynb


Age old problem
-----------------
The data format of ocean model results vary widely, and can appear on many different types of grid. In the past, GNOME accepted only specific formatting for gridded data in netCDF files, and this data was generally unavailable to other parts of the model. Adding a new data format or grid type was a difficult affair that required diving deep into the legacy C components.

Environment objects were conceptualized as a flexible and easy-to-develop representation for gridded data that would dramatically reduce the difficulty of handling the many different formats and be usable and sharable throughout the model.


Overview
--------
An important perspective to take is an abstracted view of what gridded data represents. You can imagine any gridded data as a scalar
field, where each point in space and time is associated with a value. Because the data is discrete on specific points in
space, the value of a point between these data points must be determined by some sort of interpolation.

## included in pygnome docs )Mar 2023)
An environment object implements an association between a data variable (such as a netCDF Variable, or numpy array) and a
Grid, Time, and Depth (representing the data dimensions in space and time) and does interpolation across them. By combining and/or imposing conditions on these environment objects, many natural processes can be represented. In addition, if possible, the Grid, Time, and Depth may be shared among environment objects, which provides a number of performance and programmatic benefits.
The core functionality of an environment object is it’s ‘EnvObject.at(points, time)’ function. The intent of this
function is to provide the interpolated value of the data at each point at the specified time. 

By extending and
overriding this function, more advanced behavior can be implemented. An example of this is the IceAwareCurrent,
described later in this paper.
