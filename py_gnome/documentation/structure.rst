GNOME Structure
=====================

A setup of the GNOME model consists of fairly simple structure: It is
fundamentally a Lagrangian element (particle tracking) model -- the oil or
other substance is represented as Lagrangian elements (LEs), or particles, in
the model, with their movement and properties tracked over time. The elements
are acted on by a number of "movers", each representing a different physical
process. For the most part, each mover moves the particles one way or another,
but a mover can also act to change the nature of a particle, rather than moving
it.

A GNOME setup consists of:

 * A Map
 * One or more Spills
 * One or more movers
 * Outputters
 
The Map
---------------
 
The Map in GNOME defines the domain of the model. It can consist of bounds,
shoreline (to define where land and water are), and properties of the
shoreline. For 3-d modeling, it can also define the bathymetry.
 
  
Spills
-------------
 
Spills in GNOME are a source of elements -- they define where and when the
elements are released into the model. 

Each spill is a composition of the type of substance spilled (ElementType) and
how elements are released (instantaneous, continuous, etc). Each spill is
initialized with an element type -- the element types define what properties
the elements have. The elements may represent something spilled such as an oil
spill, or could be any other object, objects or substances that you want to
track -- chemicals, floating debris, fish larvae, etc.

The ElementType itself contains 'substance' which defines the properties of
the substance spilled and a list of initializers. These are used to define
data arrays associated with the type of spill, ie floating, floating with
mass, floating weathering particles etc. There are helper functions
to define the element_type without delving into initializers; however,
user can optionally manipulate list of initializers. The helper functions are:

#. :func:`~gnome.spill.elements.element_type.floating` -
   to model floating elements with windages

#. :func:`~gnome.spill.elements.element_type.floating_mass` -
   to model floating elements with windages. This assumes the Spill contains a
   valid values for 'amount' and 'units' and it evenly distributes the total
   mass of oil spilled to each element.

#. :func:`~gnome.spill.elements.element_type.floating_weathering` -
   a helper function for defining floating weathering elements. This
   initializes an array of 'mass_components' for each element. The
   'mass_components' array represents the fraction of mass contained in each
   psuedocomponent used to model the oil/substance, since it is a mixture of
   multiple compounds. This is used for modeling weathering processes.

Currently, if weatherers are added to the model, then user **needs** to define
element_type as floating_weathering().

This maybe simplified so floating_weathering() helper is not needed -
basically, if there are any weatherers defined, then a 'mass_components' data
array will automatically be added. If you are not doing any
weathering, the Spill object uses a default helper function based on
whether a spill amount is defined or not. This should suffice for most cases.

Multiple spills can share the same substance; which can be accessed by Spill's
:meth:`~gnome.spill.Spill.get` method. Multiple spill's can also share the
same :class:`~~gnome.spill.elements.element_type.ElementType` with the
caveat that they share the same list of initializers. For example, if
two spills have different 'windages' then the two spills cannot share the
ElementType; however, they can still share the 'substance'. Currently, the
code doesn't account for spills with different substances in the same model so
it is preferred to either share the same substance or the two substances are
equal.

 
Movers
-------------
 
Movers are any physical process that moves or effects the particles. These can
be ocean currents, winds, turbulent diffusion, and weathering processes
(evaporation, etc). Each move is initialized with the the data it needs to
compute the movement. (or links to files with data in them)

The Mover API is defined so that you can write your own movers -- for instance
to model fish swimming behavior, etc. See the reference docs for the the API.

Outputters
------------

Outputters are classes use to output results. Other formats can be written, but
the currently available ones are the `Renderer` and `NetCDFOutput`

The Renderer renders a base map, and a set of transparent pngs that plot the
positions of the elements, etc. These can be composited to make a movie of the
simulation

The NetCDFOutput class outputs the element information into the netcdf file
format.
 

 
  










