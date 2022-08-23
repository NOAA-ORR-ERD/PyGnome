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

These are common variable names used for the variables PyGNOME uses. If you set the variable names in
a netcdf files to one these names, PYGNOME should be able to load the file.

**Name Mapping:**


**grid_temperature**

  Default Names: water_t,temp

  CF Standard Names: sea_water_temperature,sea_surface_temperature

**grid_salinity**

  Default Names: salt

  CF Standard Names: sea_water_salinity,sea_surface_salinity

**grid_sediment**

  Default Names: sand_06

  CF Standard Names: 

**ice_concentration**

  Default Names: ice_fraction,aice

  CF Standard Names: sea_ice_area_fraction

**bathymetry**

  Default Names: h

  CF Standard Names: depth

**grid_current**

  Default Names: u,v,w

  CF Standard Names: u,v,w

**grid_wind**

  Default Names: u,v

  CF Standard Names: u,v

**ice_velocity**

  Default Names: u,v

  CF Standard Names: u,v

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
            'u': ['air_u', 'Air_U', 'air_ucmp', 'wind_u','u-component_of_wind_height_above_ground' ],
            'v': ['air_v', 'Air_V', 'air_vcmp', 'wind_v','v-component_of_wind_height_above_ground']
        },
        'cf_names': {
            'u': ['eastward_wind', 'eastward wind'],
            'v': ['northward_wind', 'northward wind']
        }
    },
    'ice_velocity': {
        'default_names': {
            'u': ['ice_u', 'uice'],
            'v': ['ice_v', 'vice']
        },
        'cf_names': {
            'u': ['eastward_sea_ice_velocity'],
            'v': ['northward_sea_ice_velocity']
        }
    },
}

def build_table_for_docstring():
    """
    builds a table of names for the dicstring of this module

    this should be run eny time the names dict changes

    To run:

    set the working dir to this directory

    python names.py build

    The docstring for this module should be changed in place
    """
    pass

def insert_names_table(table_text):
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
    table = []
    for env_obj, names in nc_names.items():
        table.append(f"\n**{env_obj:}**\n")
        table.append(f"\n  Default Names: {','.join(names['default_names'])}\n")
        table.append(f"\n  CF Standard Names: {','.join(names['cf_names'])}\n")
    return ''.join(table)


if __name__ == "__main__":

    import os
    import sys
    import shutil

    if "rebuild" in sys.argv:
        print("rebuilding docstring")
        #table_text = build_table_for_docstring()

        insert_names_table(build_names_table())
    else:
        print("Doing Nothing")
        print('To rebuild the docstring, pass "rebuild" in on the command line:')
        print('    python names.py rebuild')



