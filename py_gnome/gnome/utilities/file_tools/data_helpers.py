"""
an assortment of utilities to help with various netcdf grid files.
"""

import netCDF4 as nc4
import pyugrid
import pysgrid
import numpy as np


def _construct_environment_objects(**kwargs):
    '''
    This function takes the arguments passed to it, and attempts to construct the appropriate
    Property object to represent it. If the argument is already a Property object or is unable
    to be parsed, it will pass through
    '''


def _init_grid(filename,
               grid_topology=None,
               dataset=None,):
    gt = grid_topology
    gf = dataset
    if gf is None:
        gf = _get_dataset(filename)
    grid = None
    if gt is None:
        try:
            grid = pyugrid.UGrid.from_nc_dataset(gf)
        except (ValueError, NameError, AttributeError):
            pass
        try:
            grid = pysgrid.SGrid.load_grid(gf)
        except (ValueError, NameError, AttributeError):
            gt = _gen_topology(filename)
    if grid is None:
        nodes = node_lon = node_lat = None
        if 'nodes' not in gt:
            if 'node_lon' not in gt and 'node_lat' not in gt:
                raise ValueError('Nodes must be specified with either the "nodes" '
                                 'or "node_lon" and "node_lat" keys')
            node_lon = gf[gt['node_lon']]
            node_lat = gf[gt['node_lat']]
        else:
            nodes = gf[gt['nodes']]
        if 'faces' in gt and gf[gt['faces']]:
            # UGrid
            faces = gf[gt['faces']]
            if faces.shape[0] == 3:
                faces = np.ascontiguousarray(np.array(faces).T - 1)
            if nodes is None:
                nodes = np.column_stack((node_lon, node_lat))
            grid = pyugrid.UGrid(nodes=nodes, faces=faces)
        else:
            # SGrid
            center_lon = center_lat = edge1_lon = edge1_lat = edge2_lon = edge2_lat = None
            if node_lon is None:
                node_lon = nodes[:, 0]
            if node_lat is None:
                node_lat = nodes[:, 1]
            if 'center_lon' in gt:
                center_lon = gf[gt['center_lon']]
            if 'center_lat' in gt:
                center_lat = gf[gt['center_lat']]
            if 'edge1_lon' in gt:
                edge1_lon = gf[gt['edge1_lon']]
            if 'edge1_lat' in gt:
                edge1_lat = gf[gt['edge1_lat']]
            if 'edge2_lon' in gt:
                edge2_lon = gf[gt['edge2_lon']]
            if 'edge2_lat' in gt:
                edge2_lat = gf[gt['edge2_lat']]
            grid = pysgrid.SGrid(node_lon=node_lon,
                                 node_lat=node_lat,
                                 center_lon=center_lon,
                                 center_lat=center_lat,
                                 edge1_lon=edge1_lon,
                                 edge1_lat=edge1_lat,
                                 edge2_lon=edge2_lon,
                                 edge2_lat=edge2_lat)
    return grid


def _gen_topology(filename,
                  dataset=None):
    '''
    Function to create the correct default topology if it is not provided

    :param filename: Name of file that will be searched for variables
    :return: List of default variable names, or None if none are found
    '''
    gf = dataset
    if gf is None:
        gf = _get_dataset(filename)
    gt = {}
    node_coord_names = [['node_lon', 'node_lat'], ['lon', 'lat'], ['lon_psi', 'lat_psi']]
    face_var_names = ['nv']
    center_coord_names = [['center_lon', 'center_lat'], ['lon_rho', 'lat_rho']]
    edge1_coord_names = [['edge1_lon', 'edge1_lat'], ['lon_u', 'lat_u']]
    edge2_coord_names = [['edge2_lon', 'edge2_lat'], ['lon_v', 'lat_v']]
    for n in node_coord_names:
        if n[0] in gf.variables.keys() and n[1] in gf.variables.keys():
            gt['node_lon'] = n[0]
            gt['node_lat'] = n[1]
            break

    if 'node_lon' not in gt:
        raise NameError('Default node topology names are not in the grid file')

    for n in face_var_names:
        if n in gf.variables.keys():
            gt['faces'] = n
            break

    if 'faces' in gt.keys():
        # UGRID
        return gt
    else:
        for n in center_coord_names:
            if n[0] in gf.variables.keys() and n[1] in gf.variables.keys():
                gt['center_lon'] = n[0]
                gt['center_lat'] = n[1]
                break
        for n in edge1_coord_names:
            if n[0] in gf.variables.keys() and n[1] in gf.variables.keys():
                gt['edge1_lon'] = n[0]
                gt['edge1_lat'] = n[1]
                break
        for n in edge2_coord_names:
            if n[0] in gf.variables.keys() and n[1] in gf.variables.keys():
                gt['edge2_lon'] = n[0]
                gt['edge2_lat'] = n[1]
                break
    return gt


def _get_dataset(filename):
    df = None
    if isinstance(filename, basestring):
        df = nc4.Dataset(filename)
    else:
        df = nc4.MFDataset(filename)
    return df
