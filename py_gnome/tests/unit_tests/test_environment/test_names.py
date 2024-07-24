"""
a couple quick tests of the names file
"""
import copy
from gnome.environment import names

example_names = {
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
}
}

def test_capitalize_name_mapping():
    name_map = copy.deepcopy(example_names)
    names.capitalize_name_mapping(name_map)


    v_names = name_map['grid_current']['default_names']['v']
    assert len(v_names) == len(set(v_names))  # no duplicates
    assert set(v_names) == set(['V_SUR', 'WATER_V', 'V', 'water_v', 'v_surface', 'v_sur', 'V_SURFACE', 'CURR_VCMP', 'curr_vcmp', 'v'])


    ic_names = name_map['ice_concentration']['default_names']
    assert len(ic_names) == len(set(ic_names))  # no duplicates
    assert set(ic_names) == set(['ICE_FRACTION', 'aice', 'ice_fraction', 'AICE'])
