import numpy as np


import os
from datetime import datetime, timedelta


from gnome import scripting
from gnome import utilities


from gnome.model import Model

from gnome.spills import point_line_release_spill
from gnome.movers import RandomMover

from gnome.environment import GridCurrent
from gnome.movers.py_current_movers import CurrentMover

from gnome.outputters import Renderer, NetCDFOutput

from gnome.environment.gridded_objects_base import Grid_S, Variable

x, y = np.mgrid[-30:30:61j, -30:30:61j]
y = np.ascontiguousarray(y.T)
x = np.ascontiguousarray(x.T)
# y += np.sin(x) / 1
# x += np.sin(x) / 5
g = Grid_S(node_lon=x,
          node_lat=y)
g.build_celltree()
t = datetime(2000, 1, 1, 0, 0)
angs = -np.arctan2(y, x)
mag = np.sqrt(x ** 2 + y ** 2)
vx = np.cos(angs) * mag
vy = np.sin(angs) * mag
vx = vx[np.newaxis, :] * 20
vy = vy[np.newaxis, :] * 20
# vx = 1/x[np.newaxis,:]
# vy = 1/y[np.newaxis,:]
# vx[vx == np.inf] = 0
# vy[vy == np.inf] = 0
# vx = vx/mag *30
# vy = vy/mag *30
# v_x = vx.copy()
# v_y - vy.copy()
# sl = [0,0:30,31:61]
# v_x = vx[:,0] * np.cos(angs) - value[:,1] * np.sin(angs)
# y = value[:,0] * np.sin(angs) + value[:,1] * np.cos(angs)
# value[:,0] = x
# value[:,1] = y

vels_x = Variable(name='v_x', units='m/s', time=[t], grid=g, data=vx)
vels_y = Variable(name='v_y', units='m/s', time=[t], grid=g, data=vy)
vg = GridCurrent(variables=[vels_y, vels_x], time=[t], grid=g, units='m/s')
point = np.zeros((1, 2))
print(vg.at(point, t))

# define base directory
base_dir = os.path.dirname(__file__)

def make_model():
    duration_hrs = 48
    time_step = 900
    num_steps = duration_hrs * 3600 / time_step
    mod = Model(start_time=t,
                duration=timedelta(hours=duration_hrs),
                time_step=time_step)

    spill = point_line_release_spill(num_elements=1000,
                                     amount=1600,
                                     units='kg',
                                     start_position=(0.5,
                                                  0.5,
                                                  0.0),
                                     release_time=t,
                                     end_release_time=t + timedelta(hours=4)
                                     )
    mod.spills += spill

    method = 'Trapezoid'
    images_dir = method + '-' + str(time_step / 60) + 'min-' + str(num_steps) + 'steps'
    renderer = Renderer(output_dir=images_dir, image_size=(800, 800))
    renderer.delay = 5
    renderer.add_grid(g)
    renderer.add_vec_prop(vg)

    renderer.graticule.set_max_lines(max_lines=0)
    mod.outputters += renderer

    mod.movers += CurrentMover(current=vg, default_num_method=method, extrapolate=True)
    mod.movers += RandomMover(diffusion_coef=10)

    netCDF_fn = os.path.join(base_dir, images_dir + '.nc')
    mod.outputters += NetCDFOutput(netCDF_fn, which_data='all')

    return mod

if __name__ == "__main__":
    scripting.make_images_dir()
    model = make_model()
    print("doing full run")
    rend = model.outputters[0]
#     rend.graticule.set_DMS(True)
    startTime = datetime.now()
    for step in model:
        if step['step_num'] == 0:
            rend.set_viewport(((-10, -10), (10, 10)))
#         if step['step_num'] == 0:
#             rend.set_viewport(((-175, 65), (-160, 70)))
        print("step: %.4i -- memuse: %fMB" % (step['step_num'],
                                              utilities.get_mem_use()))
    print(datetime.now() - startTime)
