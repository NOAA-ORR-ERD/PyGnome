Outputters
==========

Outputters allow us to save our model run results. Options include saving images at specified model time steps
or saving all the particle information into a netCDF file for further analysis.

Renderer
--------

For graphical output we add a renderer::

    from gnome.model import Model
    from gnome.outputters import Renderer
    from datetime import timedelta
    
    model = Model()
    renderer = Renderer(output_timestep=timedelta(hours=6))            
    model.outputters += renderer
    
See :class:`gnome.outputters.Renderer` for some additional parameters that can be passed to the renderer include


NetCDF Output
-------------

To save particle information into a NetCDF file we use the NetCDF outputter. In this example, we
also make use of a few utilities from the scripting package to create and clean output directories::

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

See :class:`gnome.outputters.NetCDFOutput` for some additional parameters that can be passed to the renderer include
