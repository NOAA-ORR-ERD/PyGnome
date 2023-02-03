Weatherers
==========

Processes that change the properties of the LEs are termed "weatherers" in GNOME.
For example, evaporation reduces the mass of LEs that remain floating on the water surface,
while increasing their density and viscosity.

Weathering processes modeled in GNOME include: evaporation, dispersion, dissolution,
sedimentation, emulsification, and biodegredation. (Note: as of October 2022 many
of these algorithms are still being implemented and refined -- please contact PyGNOME developers for current status).

Some examples and common use cases are shown here. For complete documentation see :mod:`gnome.weatherers` and
:mod:`gnome.environment`

Simple Example Script
---------------------

You can find a documented simple example script in:

``pygnome/py_gnome/scripts/example_scripts/weathering_script.py``

It is well commented.



Oil Definitions
---------------

In order to model weathing processes, a ``GnomeOil`` must be defined. PyGNOME comes with a small set of test oils, which can be found in: ``gnome.spills.sample_oils``. However, for most use, you will want to use a specific oil. Oil records compatible with PyGNOME's JSON format can be downloaded from:

https://adios.orr.noaa.gov

or from NOAA's public database, available from gitHub:

https://github.com/NOAA-ORR-ERD/noaa-oil-data/tree/production/data/oil

These JSON files can be loaded when specifying a spill:


.. code-block:: python

    oil = gs.GnomeOil(filename="alaska-north-slope_AD00020.json")

    spill = gs.surface_point_line_spill(num_elements=100,
                                        start_position=(-78.4, 48.2),
                                        release_time="2022-10-13T12:00:00",
                                        substance=oil,
                                        amount=100,
                                        units='bbl')


"Standard" Weathering
---------------------

If you want to use the usual full suite of weatherers, you can turn them on with one step in your model:

.. code-block:: python

    model.add_weathering()

This will add::

-Emulsification
-Evaporation
-Dispersion
-Spreading

If you want to load specific weathering modules (e.g., evaporation and dispersion), you can setup your script as follows:

.. code-block:: python

    model.add_weathering(which=('evaporation', 'dispersion'))

Note that you will need a full set of Environment objects in order for weathering to run -- see below.

Evaporation
-----------

Evaporation requires both Wind and Water Objects be initialized. Here's an example, this time reading the wind time
series from a file called "mywind.txt" (this file format is described in the :doc:`movers` section):

.. code-block:: python

    from gnome.model import Model
    from gnome.weatherers import Evaporation
    from gnome.environment import Water, Wind
    model = Model()
    wind = Wind(filename="path_2_file/mywind.txt")
    water = Water(temperature=300.0, salinity=35.0) #temperature in Kelvin, salinity in psu
    model.weatherers += Evaporation(wind=wind,water=water)

Dispersion
----------

Natural dispersion requires Wind, Water, and Waves objects are initialized.
Note that the wind is not explicitly required but is needed by the Waves object. Adding on to our example above:

.. code-block:: python

    from gnome.model import Model
    from gnome.weatherers import Evaporation, NaturalDispersion
    from gnome.environment import Water, Wind, Waves
    model = Model()
    wind = Wind(filename="path_2_file/mywind.txt")
    waves = Waves(wind)
    water = Water(temperature=300.0, salinity=35.0) #temperature in Kelvin, salinity in psu
    model.weatherers += Evaporation(wind=wind,water=water)
    model.weatherers += NaturalDispersion

Emulsification
--------------
Emulsification requires Wind and Waves objects to be initialized.
Note that the wind is not explicitly required but is needed by the Waves object. Adding on to our example above:

.. code-block:: python

    from gnome.model import Model
    from gnome.weatherers import Evaporation, NaturalDispersion
    from gnome.environment import Water, Wind, Waves
    model = Model()
    wind = Wind(filename="path_2_file/mywind.txt")
    waves = Waves(wind)
    water = Water(temperature=300.0, salinity=35.0) #temperature in Kelvin, salinity in psu
    model.weatherers += Evaporation(wind=wind,water=water)
    model.weatherers += NaturalDispersion
    model.weatherers += Emulsification(waves)

Dissolution
-----------
This module has been partially implemented in PyGNOME, but it has not been thoroughly validated and tested; therefore, it may not work as expected.

Biodegradation
--------------
This module has been partially implemented in PyGNOME, but it has not been thoroughly validated and tested; therefore, it may not work as expected.

Viewing Bulk Weathering Data
----------------------------

Since the total oil volume spilled is divided among multiple particles, bulk oil budget properties (e.g. percent of oil volume evaporated) are computed and stored in addition to the individual particle data.

These data are available through a specialized Outputter named WeatheringOutput,
see :ref:`weathering_data_output`





