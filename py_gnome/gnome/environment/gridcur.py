"""
code to work with the "old" gridcur format

Examples:

A single cell, values on the cell: ::

    [GRIDCURTIME] KNOTS
    NUMROWS 1
    NUMCOLS 1
    LOLAT 44
    HILAT 46
    LOLONG 12
    HILONG 15
    [TIME]   30 1 2002 1 0
    1 1 .092388 -.0382683

Two by two grid, values on the nodes: ::

    [GRIDCURTIME] KNOTS
    NUMROWS 2
    NUMCOLS 2
    STARTLAT 44
    STARTLONG 12
    DLAT 2
    DLONG 3
    [TIME]   30 1 2002 1 0
    1 1 .092388 -.0382683
    1 2 .092388 -.0382683
    2 1 .092388 -.0382683
    2 2 .092388 -.0382683

"""

from datetime import datetime
import numpy as np


from .gridded_objects_base import (Time, Variable, VectorVariable, Grid_R)


data_types = {"GRIDCURTIME": "currents",
              "GRIDWINDTIME": "winds",
              }
data_type_tags = {val: key for key, val in data_types.items()}

class GridCurReadError(Exception):
    """
    custom class so that we canknow the error was in reading GridCur
    file
    """
    pass



def from_gridcur(filename,
                 extrapolation_is_allowed=False,
                 **kwargs):
    """
    factory function to make a GridCurrent from a gridcur file

    :param filename: name (full path) of the gridcur file to load

    :param extrapolation_is_allowed=False:

    passes other keyword arguments on to GridCurrent initilizer
    """
    # import here to avoide circular import
    from .environment_objects import GridCurrent
    gc = GridCurrent.__new__(GridCurrent)
    init_from_gridcur(gc, filename, **kwargs)
    return gc


def init_from_gridcur(gc,
                      filename,
                      extrapolation_is_allowed=False,
                      **kwargs):
    """
    utility function to initialize a GridCurrent

    :param gc=None: the GridCur object to be initialized
                    if None, then a new one will be created.

    :param filename: name (full path) of the gridcur file to load

    :param extrapolation_is_allowed=False:

    passes other keyword arguments on to GridCurrent initilizer
    """

    # Read the file:
    data_type, units, times, lon, lat, data_u, data_v = read_file(filename)

    if (((len(lon), len(lat)) == data_u[0].shape)
        and (data_u[0].shape == data_v[0].shape)):
        location = "node"
    elif ((len(lon) - 1, len(lat) - 1) == (data_u[0].shape)
          and (data_u[0].shape == data_v[0].shape)):
        location = "face"
        raise NotImplementedError("Data on Faces not currently implemented")
    else:
        raise ValueError("There is a mismatch in the array sizes")

    grid = Grid_R(node_lon=lon, node_lat=lat)

    time = Time(data=times)

    U = Variable(
        name=f"eastward surface velocity",
        units=units,
        time=time,
        data=data_u,
        grid=grid,
        varname='u',
        location=location,
        attributes=None,
    )

    V = Variable(
        name=f"northward surface velocity",
        units=units,
        time=time,
        data=data_v,
        grid=grid,
        varname='v',
        location=location,
        attributes=None,
    )
    # total kludge, but when deserializing the name may already be there
    name = kwargs.pop('name', f"gridcur {data_type}")
    gc.__init__(name=name,
                filename=filename,
                units=units,
                time=time,
                variables=[U, V],
                varnames=('u', 'v'),
                extrapolation_is_allowed=extrapolation_is_allowed,
                **kwargs)

def read_file(filename):
    times = []
    data_u = []
    data_v = []
    grid_info = {}
    try:
        with open(filename, encoding='utf-8') as infile:
            # read the header
            for line in infile:
                # ignore lines before the header
                key = line.split()[0].strip("[]")
                if key in data_types:
                    data_type = data_types[key]
                    units = line.split()[1].strip()
                    break
            else:
                raise ValueError("No [GRIDCURTIME] or [GRIDWINDTIME] header in the file")
            # read the grid info
            for line in infile:
                if line.strip().startswith("[TIME]"):
                    break
                data = line.split()
                grid_info[data[0].strip()] = float(data[1])

            # read the data - one timestep at a time
            while True:
                if line.strip().startswith("[TIME]"):
                    time = [int(num) for num in line.split()[1:]]
                    times.append(datetime(time[2], time[1], time[0], time[3], time[4]))
                    lon, lat, U, V = make_grid_arrays(grid_info)
                    data_u.append(U)
                    data_v.append(V)
                    line = infile.readline()
                    continue
                elif not line:
                    break
                data = line.split()
                row = int(data[0]) - 1
                col = int(data[1]) - 1
                u = float(data[2])
                v = float(data[3])
                U[row, col] = u
                V[row, col] = v
                line = infile.readline()

            # put the velocities together in a single array
            data_u = np.array(data_u)
            data_v = np.array(data_v)
    # so that we will get the same error type regardless
    except Exception as ex:
        raise GridCurReadError from ex

    return data_type, units, times, lon, lat, data_u, data_v


def write_gridcur(filename, data_type, units, times, lon, lat, data_u, data_v):
    """
    write a gridcur file -- used for making tests, etc.

    """
    # there could be lots more error checking here, but why?
    if (len(lon), len(lat)) == (data_u[0].shape[0],
                                data_u[0].shape[1]):
        location = 'nodes'
    elif (len(lon), len(lat)) == ((data_u[0].shape[0] + 1),
                                  (data_u[0].shape[1] + 1)):
        location = 'cells'
    else:
        raise ValueError("shape mismatch between lat, lon, and data arrays")

    with open(filename, 'w', encoding='utf-8') as outfile:
        outfile.write(f'[{data_type_tags[data_type]}] ')
        outfile.write(f"{units}\n")
        outfile.write(f"NUMROWS {data_u[0].shape[0]}\n")
        outfile.write(f"NUMCOLS {data_u[0].shape[1]}\n")
        if location == "cells":
            outfile.write(f"LOLAT {lat[0]}\n")
            outfile.write(f"HILAT {lat[-1]}\n")
            outfile.write(f"LOLONG {lon[0]}\n")
            outfile.write(f"HILONG {lon[-1]}\n")
        elif location == "nodes":
            outfile.write(f"STARTLAT {lat[0]}\n")
            outfile.write(f"STARTLONG {lon[0]}\n")
            outfile.write(f"DLAT {(lat[-1] - lat[0]) / (len(lat) - 1)}\n")
            outfile.write(f"DLONG {(lon[-1] - lon[0]) / (len(lon) - 1)}\n")
        for time, U, V in zip(times, data_u, data_v):
            outfile.write(f"[TIME] {time.day} {time.month} {time.year} "
                          f"{time.hour} {time.minute}\n")
            for row in range(U.shape[0]):
                for col in range(U.shape[1]):
                    outfile.write(f"{row+1:4d} {col+1:4d} "
                                  f"{U[row, col]:10.6f} {V[row, col]:10.6f}\n")

def make_grid_arrays(grid_info):
    """
    Build the arrays for the grid and data

    :param grid_info: a dict of the grid information from the header
    """

    try:
        num_rows = int(grid_info["NUMROWS"])
        num_cols = int(grid_info["NUMCOLS"])
        if "LOLAT" in grid_info:  # This is a cell-centered grid
            lat = np.linspace(grid_info["LOLAT"],
                              grid_info["HILAT"],
                              int(grid_info["NUMCOLS"]) + 1)
            lon = np.linspace(grid_info["LOLONG"],
                              grid_info["HILONG"],
                              int(grid_info["NUMROWS"]) + 1)
        elif "STARTLAT" in grid_info:  # this is a node grid
            min_lat = grid_info["STARTLAT"]
            min_lon = grid_info["STARTLONG"]
            dlat = grid_info["DLAT"]
            dlon = grid_info["DLONG"]
            lat = np.linspace(min_lat, min_lat + (dlat * (num_cols - 1)), num_cols)
            lon = np.linspace(min_lon, min_lon + (dlon * (num_rows - 1)), num_rows)
    except KeyError:
        raise ValueError("File does not have full grid specification")
    U = np.zeros((num_rows, num_cols), dtype=np.float64)
    V = np.zeros((num_rows, num_cols), dtype=np.float64)

    return lon, lat, U, V

