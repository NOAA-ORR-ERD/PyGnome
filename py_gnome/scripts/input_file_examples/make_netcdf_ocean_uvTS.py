#!/usr/bin env python
"""
Creates two separate NetCDF for surface and 3D ocean currents, temperature, 
and salinity from specified model(s) within specified region and time period. 
NetCDF files are created in the same directory as the script. 

WARNING: the surface case is not as robust and can take repeated times to
    work.  When it doesn't work, errors like "NetCDF file not found" are 
    returned.  Running the script repeatidly tends to eventually lead to a 
    good connection. 

Parameters
----------
model_id : list of str
    A list of model IDs from which to extract model results.
start : datetime.datetime
    The start date and time for model results subset.
end : datetime.datetime
    The end date and time for model results subset.
bounds : tuple of float
    A tuple defining the bounding box for subsetting model results in the format 
    (west, south, east, north).
location_tag : str
    A tag to include in the output filename to indicate the location.

Returns
-------
None
    The function writes the subset of model results to NetCDF files where 
    script is called.
"""
from libgoods import api
import datetime

# user specifications
model_id = 'ESPC'
start = datetime.datetime.today()#datetime(2025, 9, 9)
end = start + datetime.timedelta(days=3)
bounds = (-88.5, 27.5, -87.5, 28.5)
location_tag = "Gulf_center28N88W"

def make_netcdf_ocean_uvTS(model_id, start, end, bounds, env_params, location_tag):
    """
    Creates NetCDF of ocean currents, temperature, and salinity from specified model(s) 
    within specified region and time period.

    Parameters
    ----------
    model_id : list of str
        A list of model IDs from which to extract model results.
    start : datetime.datetime
        The start date and time for model results subset.
    end : datetime.datetime
        The end date and time for model results subset.
    bounds : tuple of float
        A tuple defining the bounding box for subsetting model results in the format 
        (west, south, east, north).
    env_params : list of str 
        A list of environmental parameters to extract from the model results.  
        For this code, valid options are: ["surface currents", 3D currents", 
        "water properties"].  If 3D currents are selected, water properties 
        will be 3D.  "ice" and "surface winds" are also an option, but the 
        intention of this code is to return water properties.
    location_tag : str
        A tag to include in the output filename to indicate the location.

    Returns
    -------
    None
        The function writes the subset of model results to NetCDF files where 
        script is called.
    """
    
    depth = "surface" if "surface currents" in env_params else "3D"
        
    # Initialize an empty list to store the models that will be processed.
    models_to_process = []

    # Account for different formats of 'model_id' to allow for list of models
    if isinstance(model_id, list):
        models_to_process = model_id
    elif isinstance(model_id, str):
        models_to_process = [model_id]
    else:
        print("Warning: model_id should be a string or a list of strings. Skipping processing.")
        return

    # Subset (3D) model or list of models given the input criteria
    for model in models_to_process:
        print(f"Subsetting {depth} {model} currents, temp, and salinity")
        try:
            model_subset = api.get_model_subset(
                model, start, end, bounds, environmental_parameters=env_params
            )
            # now write the file
            api.get_model_output(model_subset, f'{model}_uvTS_{location_tag}_{depth}.nc')
        except Exception as e:
            print(f"Error processing model {model}: {e}")


# extract 3D currents, temperature, and salinity
env_params=['3D currents','water properties']
make_netcdf_ocean_uvTS(model_id, start, end, bounds, env_params, location_tag)

# extract 2D surface currents, temperature, and salinity
env_params=['surface currents','water properties']
make_netcdf_ocean_uvTS(model_id, start, end, bounds, env_params, location_tag)