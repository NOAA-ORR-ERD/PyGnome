GNOME Structure
=====================

A setup of the GNOME model consists of fairly simple structure: It is fundamentally a Lagrangian element (particle tracking) model -- the oil or other substance is represented as Lagrangian elements (LEs), or particles, in the model, with their movement and properties tracked over time. The LEs are acted on by a number of "movers", each representing a different physical process. For the most part, each mover moves the particles one way or another, but a mover can also act to change the nature of a particle, rather than moving it.

A GNOME setup consists of:

 * A Map
 * One or more Spills
 * One or more movers
 
The Map
---------------
 
The Map in GNOME defines the domain of the model. It can consist of bounds, shoreline (to define where land and water are), and properties of the shoreline. For 3-d modeling, it can also define the bathymetry.
 
  
Spills
-------------
 
Spills in GNOME are a collection of particles and their properties. Each spill can have a unique set of properties. A "Spill" may represent an actual spill such as an oil spill, or could be any other object, objects or substances that you want to track -- chemicals, floating debrsi, fish larvae, etc.
 
 
Movers
-------------
 
Movers are any physical process that moves or effects the particles. These can be ocean currents, winds, turbulent diffusion, and weathing processes (evaporation, etc).
 
 
  










