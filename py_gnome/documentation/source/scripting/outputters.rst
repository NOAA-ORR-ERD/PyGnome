.. _scripting_outputters:


Outputters
==========

Outputters allow you to save the model results. Options include saving images, netcdf, kml/kmz, shapefile, and geoJSON formats.

The Outputter API
-----------------

Each outputter must be initialized a little differently, depending on its capabilities and intended use, but the general API is::

    Outputter(cache=None,
              on=True,
              output_timestep=None,
              output_zero_step=True,
              output_last_step=True,
              name=None):

The ``cache`` object is where the outputter will pull its data from -- if not set, it automatically grabs the cache from the model when added to a model.

``output_timestep`` specifies when you want the output written -- it defaults to every model timestep, but you may only want output every hour, or every 6 hours, or...

``output_zero_timestep`` specifies whether you want the zeroth timestep (before any elements have been moved or altered) output.

``output_last_timestep`` specifies whether you want the final timestep output, regardless of whether it lands on a specified output_timestep.

See :class:`gnome.outputters.Outputter` for the full API.


Renderer
--------

For graphical (images) output we add a Renderer::

    import gnome.scripting as gs

    model = gs.Model()
    renderer = gs.Renderer(mapfile="the_map_bna",
                           images_dir = "output",
                           size = (1280,1024),
                           output_timestep=gs.hours(6))
    model.outputters += renderer


See :class:`gnome.outputters.Renderer` for some additional parameters that can be passed to the renderer.


NetCDF Output
-------------

To save particle information into a NetCDF file we use the NetCDF outputter. In this example, we also make use of a few utilities from the scripting package to create and clean output directories::

    import scripting as gs
    import os

    model = gs.Model()
    images_dir = 'images'
    gs.make_images_dir(images_dir)
    netcdf_file = os.path.join(images_dir, 'test_output.nc')
    gs.remove_netcdf(netcdf_file)
    nc_outputter = gs.NetCDFOutput(netcdf_file,
                                   which_data='most',
                                   output_timestep=gs.hours=(24),
                                   )
    model.outputters += nc_outputter

``which_data`` specifies what data associated with the elements you want written out to the file. the options are: 'standard', 'most' or 'all'.

'standard' : the basic stuff most people would want

'most': everything the model is tracking except the internal-use-only arrays

'all': everything tracked by the model (mostly used for diagnostics)

in addition, you can specifically set exactly which data are output by altering the ``.arrays_to_output`` attribute of the outputter. It is a python set, and data array names can be added or removed before the model is run.

The resulting netcdf files are in the "nc_particles" format. This format allows storage of arbitrary data associated with the elements, and different numbers of elements at each timestep. Code to read this format can be found here in the GnomeTools repository on gitHub:

https://github.com/NOAA-ORR-ERD/GnomeTools/blob/master/post_gnome/post_gnome/nc_particles.py

See :class:`gnome.outputters.NetCDFOutput` for some additional parameters that can be passed to the constructor to customize the output.


KMZ Output
----------

To save particle information into a KMZ file that can be read by Google Earth (and other applications), we use the KMZ outputter::

    import gnome.scripting as gs
    model.outputters += gs.KMZOutput('gnome_results.kmz',
                                     output_timestep=timedelta(hours=6))

The KMZ file contains a kml file with layers for each output timestep, uncertain and certain elements, and beached and floating elements, along with icons to render the elements.

See :class:`gnome.outputters.KMZOutput` for the full documentation


Shapefile Output
----------------

To save particle information into a Shapefile that can be read by a variety of GIS applications, we use the Shape Outputter::

    import gnome.scripting as gs
    model.outputters += ShapeOutput('gnome_results',
                                    zip_output=True,
                                    output_timestep=timedelta(hours=6))

The ShapeOutput creates a set of shapefiles, optionally all in one zip file, that contains points for the elements at each timestep, with attributes that specify the elements properties.

See :class:`gnome.outputters.ShapeOutput` for the full documentation


Oil Budget Output
-----------------

To save weathering information into a CSV file, we use the OilBudget Outputter::

    import gnome.scripting as gs
    model.outputters += OilBudgetOutput('adios_results',
                                    output_timestep=timedelta(hours=6))

The OilBudgetOutput creates a CSV file with the oil budget information (amounts in kilograms):

 * model_time
 * amount_released
 * evaporated
 * natural_dispersion
 * sedimentation
 * floating
 * beached
 * off_maps

See :class:`gnome.outputters.OilBudgetOutput` for the full documentation


.. _weathering_data_output:

Weathering Data Output
----------------------

Bulk oil budget properties (e.g. percent of total oil volume evaporated) are computed and stored in addition to the individual particle
data. These data are available through a specialized Outputter named WeatheringOutput. To save this information to a file::

    model.outputters += gs.WeatheringOutput('MyOutputDir')

Alternatively, if you want to view specific weathering information during the model run::

    model.outputters += gs.WeatheringOutput()

    for step in model:
        print "Percent evaporated is:"
        print step['WeatheringOutput']['evaporated']/step['WeatheringOutput']['amount_released'] * 100


Note: if you are running the model with a conservative or non-weathering substance, this will result in an
error as the WeatheringOutput will not contain any evaporation data. Depending on how you have set
up your model (spill substance, weatherers), WeatheringOutput may contain any or all of:

 * amount_released
 * avg_density
 * avg_viscosity
 * beached
 * dissolution
 * evaporated
 * floating
 * natural_dispersion
 * non_weathering
 * off_maps
 * sedimentation
 * time_stamp
 * water_content

See :class:`gnome.outputters.WeatheringOutput` for the full documentation
