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

The Environment Class
---------------------

As weathering processes depend on a variety of environmental conditions (e.g. wind, waves, and water properties), 
GNOME requires initialization of various environment objects before weathering processes can be added to 
the model. 

For example, to create a simple Wind object (for a constant in time wind)::

    from gnome.environment import Wind
    import numpy as np
    from datetime import datetime
    from gnome.basic_types import datetime_value_2d
    series = np.zeros((1, ),dtype=datetime_value_2d) #Make a wind time series (one value for wind constant in time)
    series[0] = (datetime(2015,1,1,0), (10, 0))
    wind = Wind(timeseries=series,units='knots')
    
More simply, we can use the helper function in :mod: `gnome.scripting` for creating a constant wind::

    from gnome.environment.wind import constant_wind
    wind = constant_wind(10,0,'knots')
    
Once environment objects are created, they can be explicitly passed to weatherers or added to the model.environment::

    from gnome.model import Model
    model = Model()
    model.environment += wind
    
If a weatherer is added to the model without explicity specifying the required environment objects, then the first object 
of the correct type in the environment collection will be used for that weathering process. For example, 
if multiple wind time series are created and added to model.environment then the first one added will be used 
for weathering processes unless explicitly specified.

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
    model.weatherers += NaturalDispersion()
