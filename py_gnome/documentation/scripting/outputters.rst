Outputters
==========

Outputters allow us to save our model run results. Options include saving images, netcdf, kmlz, and geoJSON formats.

The Outputter API
-----------------

Each outputter must be initialized a little differently, depending on its capabilities and intended use, but the general API is::

    Outputter(cache=None,
              on=True,
              output_timestep=None,
              output_zero_step=True,
              output_last_step=True,
              name=None):

The ``cache`` object is where the ouputtter will pull its data from -- if not set, it automatically grabs the cache from the model when added to a model.

``output_timestep`` specifies when you want the output written -- it defaults to every model timestep, but you may only want output every hour, or every 6 hours, or...

``output_zero_timestep`` specified whether you want the zeroth timestep (before any elements have been moved or altered) output

``output_last_timestep`` specifies whether you want the final timestep output, regardless of whether it lands on a the specified output_timestep.

See :class:`gnome.outputters.Outputter` for the full API.


Renderer
--------

For graphical (images) output we add a Renderer::

    from gnome.model import Model
    from gnome.outputters import Renderer
    from datetime import timedelta

    model = Model()
    renderer = Renderer(mapfile="the_map_bna",
                        images_dir = "output",
                        size = (1280,1024),
                        output_timestep=timedelta(hours=6))
    model.outputters += renderer


See :class:`gnome.outputters.Renderer` for some additional parameters that can be passed to the renderer.


NetCDF Output
-------------

To save particle information into a NetCDF file we use the NetCDF outputter. In this example, we also make use of a few utilities from the scripting package to create and clean output directories::

    from gnome.model import Model
    from gnome.outputters import NetCDFOutput
    from gnome import scripting
    from datetime import timedelta
    import os

    model = Model()
    images_dir = 'images'
    scripting.make_images_dir(images_dir)
    netcdf_file = os.path.join(images_dir, 'test_output.nc')
    scripting.remove_netcdf(netcdf_file)
    nc_outputter = NetCDFOutput(netcdf_file, which_data='most', output_timestep=timedelta(hours=24))
    model.outputters += nc_outputter

``which_data`` specifies what data associated with the elements you want written out to the file. the options are: 'standard', 'most' or 'all'.

'standard' : the basic stuff most people would want

'most': everything the model is tracking except the internal-use-only
        arrays

'all': everything tracked by the model (mostly used for diagnostics)

in addition, you can specifically set exactly which data are output by altering the ``.arrays_to_output`` attribute of the outputter. It is a python set, and data array names can be added or removed before the model is run.


The resulting netcdf files are in the "nc_particles" format. This format allows storage of arbitrary data associated with the elements, and different numbers of elements at each timestep. Code to read this format can be found here in the GnomeTools repository on gitHub:

https://github.com/NOAA-ORR-ERD/GnomeTools/blob/master/post_gnome/post_gnome/nc_particles.py

See :class:`gnome.outputters.NetCDFOutput` for some additional parameters that can be passed to the constructor to customize the output.


KMZ Output
----------

To save particle information into a KMZ file that can be read by Google Earth (and other applications), we use the KMZ outputter::

    model.outputters += KMZOutput('gnome_results.kmz',
                                  output_timestep=timedelta(hours=6))

The KMZ contains a kml file with layers for each output timestep, unceratain and certain elements, ans beached and floating elements, along with icons to render the elements.

See :class:`gnome.outputters.KMZOutput` for the full documentation


