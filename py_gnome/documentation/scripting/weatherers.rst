Weatherers
==========

Processes that change the properties of the LEs are termed "weatherers" in GNOME.
For example, evaporation reduces the mass of LEs that remain floating on the water surface,
while increasing their density and viscosity.

Weathering processes modeled in GNOME include: evaporation, dispersion, dissolution,
sedimentation, emulsification, and biodegredation. (Note: as of October 2015 many
of these algorithms are still being implemented and refined -- please contact GNOME developers for
current status).

Some examples and common use cases are shown here. For complete documentation see :mod:`gnome.weatherers` and
:mod:`gnome.environment`


Evaporation
-----------

Evaporation requires both Wind and Water Objects be initialized. Here's an example, this time reading the wind time
series from a file called "mywind.txt" (this file format is described in the :doc:`movers` section)::

    from gnome.model import Model
    from gnome.weatherers import Evaporation
    from gnome.environment import Water, Wind
    model = Model()
    wind = Wind(filename="path_2_file/mywind.txt")
    water = Water(temperature=300.0, salinity=35.0) #temperature in Kelvin, salinity in psu
    model.weatherers += Evaporation(wind=wind,water=water)

Dispersion
----------

Natural dispersion requires Wind, Water, and Waves objects are initialized. Note that the wind is not
explicitly required but is needed by the Waves object. Adding on to our example above::

    from gnome.model import Model
    from gnome.weatherers import Evaporation, NaturalDispersion
    from gnome.environment import Water, Wind, Waves
    model = Model()
    wind = Wind(filename="path_2_file/mywind.txt")
    waves = Waves(wind)
    water = Water(temperature=300.0, salinity=35.0) #temperature in Kelvin, salinity in psu
    model.weatherers += Evaporation(wind=wind,water=water)
    model.weatherers += NaturalDispersion


Dissolution
-----------

Emulsification
--------------

Biodegradation
--------------

Viewing Bulk Weathering Data
----------------------------

Since the total oil volume spilled is divided among multiple particles, bulk oil budget properties
(e.g. percent of oil volume evaporated) are computed and stored in addition to the individual particle
data. These data are available through a specialized Outputter named WeatheringOutput,
see :ref:`weathering_data_output`





