"""
Module that provides the names used when processing netcdf files

Each type of Environment object has a set of names that it uses
to determine what the contents of a netcdf file are.

**cf_names:**

These are "standard names", as defined by the CF metadata standard:

https://cfconventions.org/standard-names.html

These are the best options to use, as the are standardized. When loading a netcdf file, the Environment
objects will first look for variable with cf_names, to identify their meaning. If no
variables exist with the standard names, then common variable names will be used.

**nc_names:**

These are common variable names used for the variables PyGNOME uses.
If you set the variable names in a netcdf files to one these names,
PyGNOME should be able to load the file.

**Name Mapping:**


**grid_temperature**

  Default Names: temp, water_t, TEMP, WATER_T


  CF Standard Names: sea_water_temperature, sea_surface_temperature


**grid_salinity**

  Default Names: salt, SALT


  CF Standard Names: sea_water_salinity, sea_surface_salinity


**grid_sediment**

  Default Names: sand_06, SAND_06


  CF Standard Names: 


**ice_concentration**

  Default Names: aice, ice_fraction, AICE, ICE_FRACTION


  CF Standard Names: sea_ice_area_fraction


**bathymetry**

  Default Names: H, h


  CF Standard Names: depth


**grid_current**

 Default Names for u: water_u, u_sur, WATER_U, CURR_UCMP, u_surface, u, U_SUR, U_SURFACE, U, curr_ucmp

 Default Names for v: V, curr_vcmp, v_surface, v_sur, CURR_VCMP, V_SUR, v, water_v, WATER_V, V_SURFACE

 Default Names for w: W, w


 CF Standard Names for u: eastward_sea_water_velocity, surface_eastward_sea_water_velocity

 CF Standard Names for v: northward_sea_water_velocity, surface_northward_sea_water_velocity

 CF Standard Names for w: upward_sea_water_velocity


**grid_wind**

 Default Names for u: AIR_UCMP, wind_u, air_ucmp, Air_U, WIND_U, AIR_U, UWIND, air_u, U-COMPONENT_OF_WIND_HEIGHT_ABOVE_GROUND, u-component_of_wind_height_above_ground, u10, Uwind, U10

 Default Names for v: Air_V, air_vcmp, VWIND, v10, V10, air_v, AIR_VCMP, V-COMPONENT_OF_WIND_HEIGHT_ABOVE_GROUND, WIND_V, wind_v, v-component_of_wind_height_above_ground, Vwind, AIR_V


 CF Standard Names for u: eastward_wind, eastward wind

 CF Standard Names for v: northward_wind, northward wind


**ice_velocity**

 Default Names for u: ice_u, UUICE, uice, uuice, ICE_U, UICE

 Default Names for v: VICE, vice, ICE_V, VVICE, ice_v, vvice


 CF Standard Names for u: eastward_sea_ice_velocity

 CF Standard Names for v: northward_sea_ice_velocity

"""

nc_names = {
    'grid_temperature': {
        'default_names': ['water_t', 'temp'],
        'cf_names': ['sea_water_temperature', 'sea_surface_temperature']
    },
    'grid_salinity': {
        'default_names': ['salt'],
        'cf_names': ['sea_water_salinity', 'sea_surface_salinity']
    },
    'grid_sediment': {
        'default_names': ['sand_06'],
        'cf_names': []
    },
    'ice_concentration': {
        'default_names': ['ice_fraction', 'aice'],
        'cf_names': ['sea_ice_area_fraction']
    },
    'bathymetry': {
        'default_names': ['h'],
        'cf_names': ['depth']
    },
    'grid_current': {
        'default_names': {
            'u': ['u', 'U', 'water_u', 'curr_ucmp', 'u_surface', 'u_sur'],
            'v': ['v', 'V', 'water_v', 'curr_vcmp', 'v_surface', 'v_sur'],
            'w': ['w', 'W']
        },
        'cf_names': {
            'u': [
                'eastward_sea_water_velocity',
                'surface_eastward_sea_water_velocity'
            ],
            'v': [
                'northward_sea_water_velocity',
                'surface_northward_sea_water_velocity'
            ],
            'w': ['upward_sea_water_velocity']
        }
    },
    'grid_wind': {
        'default_names': {
            'u': ['air_u', 'Air_U', 'air_ucmp', 'wind_u','u-component_of_wind_height_above_ground','Uwind','u10'],
            'v': ['air_v', 'Air_V', 'air_vcmp', 'wind_v','v-component_of_wind_height_above_ground','Vwind','v10']
        },
        'cf_names': {
            'u': ['eastward_wind', 'eastward wind'],
            'v': ['northward_wind', 'northward wind']
        }
    },
    'ice_velocity': {
        'default_names': {
            'u': ['ice_u', 'uice', 'uuice'],
            'v': ['ice_v', 'vice', 'vvice']
        },
        'cf_names': {
            'u': ['eastward_sea_ice_velocity'],
            'v': ['northward_sea_ice_velocity']
        }
    },
}

def capitalize_name_mapping(name_map):
    """
    This function will go through the name mapping, and add capitalized
    versions of all  the default_names

    NOTE: the name_map is changed in place.
    """
    for variable, data in name_map.items():
        def_names = data['default_names']
        try:
            for sub_var in def_names.values():
                for name in sub_var[:]:
                    sub_var.append(name.upper())
                    # eliminate the duplicates
                sub_var[:] = list(set(sub_var))
        except AttributeError:  # it's not nested
            for name in def_names[:]:
                def_names.append(name.upper())
            def_names[:] = list(set(def_names))


capitalize_name_mapping(nc_names)

def insert_names_table(table_text):
    """
    function to insert the names table into the docstring

    should be run when name mapping is updated.
    """
    this = __file__
    tempfile = "names.temp.py"
    with open(this) as infile:
        contents = iter(infile.readlines())


    with open(tempfile, 'w') as outfile:
        for line in contents:
            outfile.write(line)
            if "**Name Mapping:**" in line:
               break
        outfile.write("\n")
        outfile.write(table_text)
        outfile.write("\n")
        for line in contents:
            if line.strip() == '"""':
                outfile.write(line)
                break
        for line in contents:
            outfile.write(line)

    shutil.copy(tempfile, this)
    os.remove(tempfile)


def build_names_table():
    """
    This builds the table of names for the docstring (and the docs)

    NOTE: it could use some fancier rst formatting ...
    """
    table = []
    for env_obj, names in nc_names.items():
        table.append(f"\n\n**{env_obj:}**\n")
        try:  # some are dicts, some are lists ...
            for var, var_names in names['default_names'].items():
                table.append(f"\n Default Names for {var}: {', '.join(var_names)}\n")
        except AttributeError:
            table.append(f"\n  Default Names: {', '.join(names['default_names'])}\n")
        table.append('\n')
        try:  # some are dicts, some are lists ...
            for var, var_names in names['cf_names'].items():
                table.append(f"\n CF Standard Names for {var}: {', '.join(var_names)}\n")
        except AttributeError:
            table.append(f"\n  CF Standard Names: {', '.join(names['cf_names'])}\n")
    return ''.join(table)


if __name__ == "__main__":

    """
    when run as a script, builds a table of names for the docstring of this module

    this should be run any time the names dict changes

    To run:

    set the working dir to this directory

    python names.py build

    The docstring for this module should be changed in place
    """

    import os
    import sys
    import shutil

    if "rebuild" in sys.argv:
        print("rebuilding docstring")
        insert_names_table(build_names_table())
    else:
        print("Doing Nothing")
        print('To rebuild the docstring, pass "rebuild" in on the command line:')
        print('    python names.py rebuild')
