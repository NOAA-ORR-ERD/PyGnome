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

data_types = {"GRIDCURTIME": "currents",
              "GRIDWINDTIME": "winds",
              }

def read_file(filename):
    times = []
    data_u = []
    data_v = []
    grid_info = {}
    with open(filename, encoding='utf-8') as infile:
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
                time = [int(num) for num in line.split()[1:]]
                times.append(datetime(time[2], time[1], time[0], time[3], time[4]))
                break
            else:
                data = line.split()
                grid_info[data[0].strip()] = int(data[1])
        # read the data
        lon, lat, U, V = make_grid_arrays(grid_info)
        for line in infile:
            data = line.split()
            row = int(data[0]) - 1
            col = int(data[1]) - 1
            u = float(data[2])
            v = float(data[3])
            U[row, col] = u
            V[row, col] = v
        data_u.append(U)
        data_v.append(V)
        # need to read other timesteps!
        return data_type, times, data_u, data_v


def make_grid_arrays(grid_info):
    """
    build the arrays for the grid and data

    :param grid_info: a dict of the grid information from the header
    """

    try:
        num_rows = grid_info["NUMROWS"]
        num_cols = grid_info["NUMCOLS"]
        if "LOLAT" in grid_info:  # This is a cell-centered grid
            lat = np.linspace(grid_info["LOLAT"],
                              grid_info["HILAT"],
                              grid_info["NUMROWS"] + 1)
            lon = np.linspace(grid_info["LOLONG"],
                              grid_info["HILONG"],
                              grid_info["NUMCOLS"] + 1)
        elif "STARTLAT" in grid_info:  # this is a node grid
            min_lat = grid_info["STARTLAT"]
            min_lon = grid_info["STARTLONG"]
            dlat = grid_info["DLAT"]
            dlon = grid_info["DLONG"]
            lat = np.linspace(min_lat, min_lat + (dlat * num_cols), num_cols)
            lon = np.linspace(min_lon, min_lon + (dlon * num_rows), num_rows)
    except KeyError:
        raise ValueError("File does not have full grid specification")
    U = np.zeros((num_rows, num_cols), dtype = np.float64)
    V = np.zeros((num_rows, num_cols), dtype = np.float64)

    return lon, lat, U, V






