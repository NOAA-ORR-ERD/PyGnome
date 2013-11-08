GNOME Structure
=====================

A setup of the GNOME model consists of fairly simple structure: It is fundamentally a Lagrangian element (particle tracking) model -- the oil or other substance is represented as Lagrangian elements (LEs), or particles, in the model, with their movement and properties tracked over time. The elements are acted on by a number of "movers", each representing a different physical process. For the most part, each mover moves the particles one way or another, but a mover can also act to change the nature of a particle, rather than moving it.

A GNOME setup consists of:

 * A Map
 * One or more Spills
 * One or more movers
 * Outputters
 
The Map
---------------
 
The Map in GNOME defines the domain of the model. It can consist of bounds, shoreline (to define where land and water are), and properties of the shoreline. For 3-d modeling, it can also define the bathymetry.
 
  
Spills
-------------
 
Spills in GNOME are a source of elements -- they define where and when the elements are released into the model. Each spill is initialized with an element type -- teh element tpes define what properties the elements have. The elements may represent something spilled such as an oil spill, or could be any other object, objects or substances that you want to track -- chemicals, floating debris, fish larvae, etc.
 
 
Movers
-------------
 
Movers are any physical process that moves or effects the particles. These can be ocean currents, winds, turbulent diffusion, and weathering processes (evaporation, etc). Each move is initialized with the the data it needs to compute the movement. (or links to files with data in them)

The Mover API is defined so that you can write your own movers -- for instance to model fish swimming behavior, etc. See the reference docs for the the API.

Outputters
------------

Outputters are classes use to output results. Other formats can be written, but the currently available ones are the `Renderer` and `NetCDFOutput`

The Renderer renders a base map, and a set of transparent pngs that plot the positions of the elements, etc. These can be composited to make a movie of the simulation

The NetCDFOutput class outputs the element information into the netcdf file format.
 

 
  










