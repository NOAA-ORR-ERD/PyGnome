import numpy as np
import netCDF4 as nc4

from pysgrid import SGrid
from gnome.environment.grid_property import GriddedProp

import os
from datetime import datetime, timedelta


from gnome import scripting
from gnome import utilities


from gnome.model import Model

from gnome.spill import point_line_release_spill
from gnome.movers import RandomMover, constant_wind_mover, GridCurrentMover

from gnome.environment import GridCurrent
from gnome.environment import PyGrid, PyGrid_U
from gnome.movers.py_current_movers import PyCurrentMover

from gnome.outputters import Renderer, NetCDFOutput

def gen_vortex_3D(filename=None):
    x, y = np.mgrid[-30:30:61j, -30:30:61j]
    y = np.ascontiguousarray(y.T)
    x = np.ascontiguousarray(x.T)
    x_size = 61
    y_size = 61
    g = PyGrid(node_lon=x,
               node_lat=y)
    g.build_celltree()
    lin_nodes = g._trees['node'][1]
    lin_faces = np.array([np.array([([lx, lx + x_size + 1, lx + 1], [lx, lx + x_size, lx + x_size + 1]) for lx in range(0, x_size - 1, 1)]) + ly * x_size for ly in range(0, y_size - 1)])
    lin_faces = lin_faces.reshape(-1, 3)
    # y += np.sin(x) / 1
    # x += np.sin(x) / 5

    t0 = datetime(2001, 1, 1, 0, 0)
    tarr = [t0 + timedelta(hours=i) for i in range(0, 11)]
    angs = -np.arctan2(y, x)
    mag = np.sqrt(x ** 2 + y ** 2)
    vx = np.cos(angs) * mag
    vy = np.sin(angs) * mag
    vx = vx[np.newaxis, :] * 20
    vy = vy[np.newaxis, :] * 20
    vw = -0.001

    d_scale = [1, 0.5, 0, -0.5, -1]
    t_scale = np.linspace(0, 1, 11)

    tvx = np.array([vx * t for t in t_scale]).squeeze()
    tvy = np.array([vy * t for t in t_scale]).squeeze()

    dvx = np.array([vx * s for s in d_scale]).squeeze()
    dvy = np.array([vy * s for s in d_scale]).squeeze()

    tdvx = np.array([dvx * t for t in t_scale]).squeeze()
    tdvy = np.array([dvy * t for t in t_scale]).squeeze()

    lin_vx = vx.reshape(-1)
    lin_vy = vy.reshape(-1)

    lin_tvx = np.array([lin_vx * t for t in t_scale])
    lin_tvy = np.array([lin_vy * t for t in t_scale])

    lin_dvx = np.array([lin_vx * s for s in d_scale])
    lin_dvy = np.array([lin_vy * s for s in d_scale])

    lin_tdvx = np.array([lin_dvx * t for t in t_scale])
    lin_tdvy = np.array([lin_dvy * t for t in t_scale])
    
    ds = None
    if filename is not None:
        ds = nc4.Dataset(filename, 'w', diskless=True, persist=True)
        ds.createDimension('y', y.shape[0])
        ds.createDimension('x', x.shape[1])
        ds.createDimension('time', len(tarr))
        ds.createDimension('depth', len(d_scale))
        ds.createVariable('x', 'f8', dimensions=('x', 'y'))
        ds['x'][:] = x
        ds.createVariable('y', 'f8', dimensions=('x', 'y'))
        ds['y'][:] = y
        ds.createVariable('time', 'f8', dimensions=('time'))
        ds['time'][:] = nc4.date2num(tarr, 'hours since {0}'.format(t0))
        ds['time'].setncattr('units', 'hours since {0}'.format(t0))
        ds.createVariable('vx', 'f8', dimensions=('x', 'y'))
        ds.createVariable('vy', 'f8', dimensions=('x', 'y'))
        ds['vx'][:] = vx
        ds['vy'][:] = vy
        ds.createVariable('tvx', 'f8', dimensions=('time', 'x', 'y'))
        ds.createVariable('tvy', 'f8', dimensions=('time', 'x', 'y'))
        ds['tvx'][:] = tvx
        ds['tvy'][:] = tvy
        ds.createVariable('dvx', 'f8', dimensions=('depth', 'x', 'y'))
        ds.createVariable('dvy', 'f8', dimensions=('depth', 'x', 'y'))
        ds['dvx'][:] = dvx
        ds['dvy'][:] = dvy
        ds.createVariable('tdvx', 'f8', dimensions=('time', 'depth', 'x', 'y'))
        ds.createVariable('tdvy', 'f8', dimensions=('time', 'depth', 'x', 'y'))
        ds['tdvx'][:] = tdvx
        ds['tdvy'][:] = tdvy
        for v in ds.variables:
            if 'v' in v:
                ds[v].units = 'm/s'
        
        ds.createDimension('nv', lin_nodes.shape[0])
        ds.createDimension('nele', lin_faces.shape[0])
        ds.createDimension('two', 2)
        ds.createDimension('three', 3)
        ds.createVariable('nodes', 'f8', dimensions=('nv', 'two'))
        ds.createVariable('faces', 'f8', dimensions=('nele', 'three'))
        ds.createVariable('lin_vx', 'f8', dimensions=('nv'))
        ds.createVariable('lin_vy', 'f8', dimensions=('nv'))
        ds.createVariable('lin_tvx', 'f8', dimensions=('time', 'nv'))
        ds.createVariable('lin_tvy', 'f8', dimensions=('time', 'nv'))
        ds.createVariable('lin_dvx', 'f8', dimensions=('depth', 'nv'))
        ds.createVariable('lin_dvy', 'f8', dimensions=('depth', 'nv'))
        ds.createVariable('lin_tdvx', 'f8', dimensions=('time', 'depth', 'nv'))
        ds.createVariable('lin_tdvy', 'f8', dimensions=('time', 'depth', 'nv'))
        for k, v in {'nodes': lin_nodes,
                     'faces': lin_faces,
                     'lin_vx': lin_vx,
                     'lin_vy': lin_vy,
                     'lin_tvx': lin_tvx,
                     'lin_tvy': lin_tvy,
                     'lin_dvx': lin_dvx,
                     'lin_dvy': lin_dvy,
                     'lin_tdvx': lin_tdvx,
                     'lin_tdvy': lin_tdvy
                     }.items():
            ds[k][:] = v
            if 'lin' in k:
                ds[k].units = 'm/s'
        PyGrid._get_grid_type(ds, grid_topology={'node_lon': 'x', 'node_lat': 'y'})
        PyGrid._get_grid_type(ds)
        ds.setncattr('grid_type', 'sgrid')
    if ds is not None:
        # Need to test the dataset...
        from gnome.environment import GridCurrent
        from gnome.environment.grid_property import GriddedProp
        sgt = {'node_lon': 'x', 'node_lat': 'y'}
        sg = PyGrid.from_netCDF(dataset=ds, grid_topology=sgt, grid_type='sgrid')
        sgc1 = GridCurrent.from_netCDF(dataset=ds, varnames=['vx', 'vy'], grid_topology=sgt)
        sgc2 = GridCurrent.from_netCDF(dataset=ds, varnames=['tvx', 'tvy'], grid_topology=sgt)
        sgc3 = GridCurrent.from_netCDF(dataset=ds, varnames=['dvx', 'dvy'], grid_topology=sgt)
        sgc4 = GridCurrent.from_netCDF(dataset=ds, varnames=['tdvx', 'tdvy'], grid_topology=sgt)
        
        ugt = {'nodes': 'nodes', 'faces': 'faces'}
#         ug = PyGrid_U(nodes=ds['nodes'][:], faces=ds['faces'][:])
        ugc1 = GridCurrent.from_netCDF(dataset=ds, varnames=['lin_vx', 'lin_vy'], grid_topology=ugt)
        ugc2 = GridCurrent.from_netCDF(dataset=ds, varnames=['lin_tvx', 'lin_tvy'], grid_topology=ugt)
        ugc3 = GridCurrent.from_netCDF(dataset=ds, varnames=['lin_dvx', 'lin_dvy'], grid_topology=ugt)
        ugc4 = GridCurrent.from_netCDF(dataset=ds, varnames=['lin_tdvx', 'lin_tdvy'], grid_topology=ugt)
        
        ds.close()
    return {'sgrid': (x, y),
            'sgrid_vel': (dvx, dvy),
            'sgrid_depth_vel': (tdvx, tdvy),
            'ugrid': (lin_nodes, lin_faces),
            'ugrid_depth_vel': (lin_tdvx, lin_tdvy)}


def gen_sinusoid(filename=None):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', zlim=[-2, 2], xlim=[0, 25], ylim=[-2, 2])
    ax.autoscale(False)

#     import matplotlib.pyplot as plt
#     import numpy as np
#     import math
#     fig = plt.figure()
#     ax = fig.add_subplot(111)

    y, x = np.mgrid[-1:1:5j, 0:(6 * np.pi):25j]
    y = y + np.sin(x / 2)
    Z = np.zeros_like(x)
    # abs(np.sin(x / 2)) + 
    vx = np.ones_like(x)
    vy = np.cos(x / 2) / 2
    vz = np.zeros_like(x)
    ax.plot_wireframe(x, y, Z, rstride=1, cstride=1, color='blue')
    ax.quiver(x, y, Z, vx, vy, vz, length=0.5, arrow_length_ratio=0.2, color='darkblue', pivot='tail')
#     ax.quiver(x, y, vx, vy, color='darkblue', pivot='tail', angles='xy', scale=1.5, scale_units='xy', width=0.0025)
#     ax.plot(x[2], y[2])
    rho = {'r_grid': (x, y, Z),
           'r_vel': (vx, vy, vz)}

    yc, xc = np.mgrid[-0.75:0.75: 4j, 0.377:18.493:24j]
    yc = yc + np.sin(xc / 2)
    zc = np.zeros_like(xc) + 0.025
    vxc = np.ones_like(xc)
    vyc = np.cos(xc / 2) / 2
    vzc = np.zeros_like(xc)
    ax.plot_wireframe(xc, yc, zc, rstride=1, cstride=1, color="red")
    ax.quiver(xc, yc, zc, vxc, vyc, vzc, length=0.3, arrow_length_ratio=0.2, color='darkred', pivot='tail')
    psi = {'p_grid': (xc, yc, zc),
           'p_vel': (vxc, vyc, vzc)}

    yu, xu = np.mgrid[-1:1:5j, 0.377:18.493:24j]
    yu = yu + np.sin(xu / 2)
    zu = np.zeros_like(xu) + 0.05
    vxu = np.ones_like(xu) * 2
    vyu = np.zeros_like(xu)
    vzu = np.zeros_like(xu)
    ax.plot_wireframe(xu, yu, zu, rstride=1, cstride=1, color="purple")
    ax.quiver(xu, yu, zu, vxu, vyu, vzu, length=0.3, arrow_length_ratio=0.2, color='indigo', pivot='tail')
    u = {'u_grid': (xu, yu, zu),
         'u_vel': (vzu, vxu, vzu)}

    yv, xv = np.mgrid[-0.75:0.75: 4j, 0:18.87:25j]
    yv = yv + np.sin(xv / 2)
    zv = np.zeros_like(xv) + 0.075
    vxv = np.zeros_like(xv)
    vyv = np.cos(xv / 2) / 2
    vzv = np.zeros_like(xv)
    ax.plot_wireframe(xv, yv, zv, rstride=1, cstride=1, color="y")
    ax.quiver(xv, yv, zv, vxv, vyv, vzv, length=0.3, arrow_length_ratio=0.2, color='olive', pivot='tail')
    v = {'v_grid': (xv, yv, zv),
         'v_vel': (vyv, vzv, vzv)}

    angle = np.cos(x / 2) / 2

    ds = None
    if filename is not None:
        ds = nc4.Dataset(filename, 'w', diskless=True, persist=True)
        for k, v in {'eta_psi': 24,
                     'xi_psi': 4,
                     'eta_rho': 25,
                     'xi_rho':5}.items():
            ds.createDimension(k, v)
        for k, v in {'lon_rho': ('xi_rho', 'eta_rho', x),
                     'lat_rho': ('xi_rho', 'eta_rho', y),
                     'lon_psi': ('xi_psi', 'eta_psi', xc),
                     'lat_psi': ('xi_psi', 'eta_psi', yc),
                     'lat_u': ('xi_rho', 'eta_psi', xu),
                     'lon_u': ('xi_rho', 'eta_psi', yu),
                     'lat_v': ('xi_psi', 'eta_rho', xv),
                     'lon_v': ('xi_psi', 'eta_rho', yv),
                     }.items():
            ds.createVariable(k, 'f8', dimensions=v[0:2])
            ds[k][:] = v[2]
        for k, v in {'u_rho': ('xi_rho', 'eta_rho', vx),
                     'v_rho': ('xi_rho', 'eta_rho', vy),
                     'u_psi': ('xi_psi', 'eta_psi', vxc),
                     'v_psi': ('xi_psi', 'eta_psi', vyc),
                     'u': ('xi_rho', 'eta_psi', vxu),
                     'v': ('xi_psi', 'eta_rho', vyv)}.items():
            ds.createVariable(k, 'f8', dimensions=v[0:2])
            ds[k][:] = v[2]
            ds[k].units = 'm/s'
        ds.grid_type = 'sgrid'
        ds.createVariable('angle', 'f8', dimensions=('xi_rho', 'eta_rho'))
        ds['angle'][:] = angle
    if ds is not None:
        # Need to test the dataset...
        from gnome.environment import GridCurrent
        sg = PyGrid.from_netCDF(dataset=ds)
        sgc1 = GridCurrent.from_netCDF(dataset=ds, varnames=['u_rho', 'v_rho'], grid=sg)
        sgc1.angle = None
        sgc2 = GridCurrent.from_netCDF(dataset=ds, varnames=['u_psi', 'v_psi'], grid=sg)
        sgc2.angle = None
        sgc3 = GridCurrent.from_netCDF(dataset=ds, grid=sg)
        
        ds.close()

#     plt.show()


def gen_ring(filename=None):
    import matplotlib.pyplot as plt
    import matplotlib.tri as tri
    import math

    n_angles = 36
    n_radii = 6
    min_radius = 0.25
    radii = np.linspace(min_radius, 1, n_radii)

    angles = np.linspace(0, 2 * math.pi, n_angles, endpoint=False)
    angles = np.repeat(angles[..., np.newaxis], n_radii, axis=1)
    angles[:, 1::2] += math.pi / n_angles
    print angles.shape

    x = (radii * np.cos(angles)).flatten()
    y = (radii * np.sin(angles)).flatten()
    z = (np.cos(radii) * np.cos(angles * 3.0)).flatten()

    # Create the Triangulation; no triangles so Delaunay triangulation created.
    triang = tri.Triangulation(x, y)
    # Mask off unwanted triangles.
    xmid = x[triang.triangles].mean(axis=1)
    ymid = y[triang.triangles].mean(axis=1)
    mask = np.where(xmid * xmid + ymid * ymid < min_radius * min_radius, 1, 0)
    triang.set_mask(mask)
    faces = triang.get_masked_triangles()

    vx = (np.ones_like(radii) * np.cos(angles + np.pi)).flatten()
    vy = (np.ones_like(radii) * np.sin(angles)).flatten()

    if filename is not None:
        ds = nc4.Dataset(filename, 'w', diskless=True, persist=True)
        ds.createDimension('nv', x.shape[0])
        ds.createDimension('nele', faces.shape[0])
        ds.createDimension('three', 3)
        for k, v in {'node_lon': ('nv', x),
                     'node_lat': ('nv', y),
                     'faces': ('nele', 'three', faces),
                     'u': ('nv', vx),
                     'v': ('nv', vy)}.items():
            ds.createVariable(k, 'f8', dimensions=v[0:-1])
            ds[k][:] = v[-1]
            ds[k].units = 'm/s'

    if ds is not None:
        gc = GridCurrent.from_netCDF(dataset=ds)
        print gc.grid.node_lon.shape
        print gc.grid.faces.shape

    # tripcolor plot.
    plt.figure()
    plt.gca().set_aspect('equal')
    plt.triplot(triang, 'bo-')
    plt.quiver(x, y, vy, vx)
    plt.title('triplot of Delaunay triangulation')


def gen_all(path=None):
    filenames = ['staggered_sine_channel.nc', '3D_circular.nc', 'tri_ring.nc']
    if path is not None:
        filenames = [os.path.join(path, fn) for fn in filenames]
    for fn, func in zip(filenames, (gen_sinusoid, gen_vortex_3D, gen_ring)):
        func(fn)

if __name__ == '__main__':
    gen_sinusoid('staggered_sine_channel.nc')
    gen_vortex_3D('3D_circular.nc')
    gen_ring('tri_ring.nc')
