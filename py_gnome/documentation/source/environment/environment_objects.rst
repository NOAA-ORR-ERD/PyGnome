Environment Objects
===================

Environment objects are  designed to accomplish the following objectives:
 - Provide easy-to-create representations of compatible data
 - Allow a reasonably Python-literate PyGNOME user to create a PyGNOME-compatible representation of
   non-standard gridded data without having to resort to reformatting their data source.
 - Provide functions that make working with gridded data convenient, such as interpolation of data,
   automatic vector rotation, etc.
 - Allow a skilled PyGNOME user to easily create new environment objects that represent more nuanced
   phenomena, such as a surface current that is affected by ice coverage, or a wind that is sensitive to
   locations of tidal flats.
 - Whenever possible, accomplish data representation with objects that share a generic, interchangable API
 - Allow and encourage sharing of common resources among environment objects to increase simulation performance


Environment objects are designed to represent a natural phenomenon and provide an interface that can be queried in time and space. These objects
can represent a space-independent time series or gridded, time dependent data. They can also represent scalar and (2D) vector phenomena.
They are the core means for representing natural phenomena in the latest versions of PyGNOME.

Examples of things that environment objects can represent include: temperature, water velocity, wind speed & direction time series, etc.

For documentation of the API and implemented objects see :mod:`gnome.environment.environment_objects`

Background
----------

The environmental models in GNOME are often driven with data created by other models, such as ROMS, HYCOM, etc. 
In the past, the output from
these models were processed by renaming and regridding to conform to GNOME's expectations before use. Regridding every new dataset is inconvenient,
and also immediately introduces inaccuracies into the representation of reality.

Scalar or Vector
-----------------

All environment objects represent either scalar or vector data. All environment objects that represent vector
data are composed of environment objects representing scalar data. However, some more advanced vector objects
can be composed of other vector objects and use them in various custom ways


Shared Components and Memoization
---------------------------------

.. important::
   This is a performance-critical feature! Understanding and usage is highly recommended

A key design goal was to allow two objects to share common components. For example, if a user has a data file containing
temperature and salinity on the same grid, we likely do not want to create two separate representations of the grid, time, etc to
attach to the objects. As noted above, this is a key performance optimization that allows these objects to efficiently
represent the data within a file. By sharing components in this manner, memoization allows large performance increases
when sets of query points do not change between every function call.

Consider the following situation. Assume you have 100,000 points, and you want to find the water velocity
at all of them, using linear interpolation between data points. Water velocity is represented by a GridCurrent,
which in turn is represented by a *u-component* and *v-component*

1. To determine the *u-component* at any given point, you need to find which cell the point is in.
2. Then you compute the interpolation weights from the point to the nodes of the cell containig the point.
3. Finally, you multiply the values at the nodes with the interpolation alphas & sum to get the final value

- Note that all of the above operations are vectorized over the 100,000 points.

However, this only gets you the *u-component* of the result. The above computations must be repeated for the
*v-component*. Parts 1 and 2 are the responsibility of the grid component of each object. The grid object's
interpolation alpha and point location functions both implement result memoization based on a hash of the
array of points provided. If the *u-component* and *v-component* share a grid object, it will be able to
recognize when it receives the same set of points again, and therefore return the same answer for parts 1 and 2,
allowing computation to skip directly to part 3.

This combination of sharing and memoization is key to efficient composition of environment objects without
requiring custom results aggregation code for every new combination. Consider the operations required to
interpolate N variables to P points without memoization::

    ops = N*(P*locate_points + P*interpolation_alphas + P*multiply&sum)

With memoization, the equation is as follows::

    ops = P*locate_points + P*interpolation_alphas + N*P*multiply&sum + N*hash(P)

Considering locate_points and interpolation_alphas are much more computationally expensive than multiply&sum and hash, this is
a dramatic performance gain for even N=2

.. Examples in a Jupyter Notebook:
.. ...............................

.. .. toctree::
..   :maxdepth: 2

..   examples.ipynb



