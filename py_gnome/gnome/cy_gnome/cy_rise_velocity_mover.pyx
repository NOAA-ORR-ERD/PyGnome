cimport numpy as cnp
import numpy as np

# following exist in gnome.cy_gnome
from type_defs cimport *
from movers cimport RiseVelocity_c, Mover_c, get_rise_velocity
cimport cy_mover

"""
Dynamic casts are not currently supported in Cython - define it here instead.
Since this function is custom for each mover, just keep it with the definition
for each mover
"""
cdef extern from *:
    RiseVelocity_c* dynamic_cast_ptr "dynamic_cast<RiseVelocity_c *>" (Mover_c *) except NULL

cdef class CyRiseVelocityMover(cy_mover.CyMover):

    cdef RiseVelocity_c *rise_vel

    def __cinit__(self):
        self.mover = new RiseVelocity_c()
        self.rise_vel = dynamic_cast_ptr(self.mover)

    def __dealloc__(self):
        del self.mover
        self.rise_vel = NULL

#    def __init__(self, water_density=1020, water_viscosity=.000001):
    def __init__(self):
        """
        Default water_density = 1020 [kg/m3]
        Default water_viscosity = 1e-6
        """
#         if water_density <= 0:
#             raise ValueError("CyRiseVelocityMover: water_density must be >= 0")
#         if water_viscosity <= 0:
#             raise ValueError("CyRiseVelocityMover: water_viscosity must"
#                              " be >= 0")
# 
#         self.rise_vel.water_density = water_density
#         self.rise_vel.water_viscosity = water_viscosity
# 
#     property water_density:
#         def __get__(self):
#             return self.rise_vel.water_density
# 
#         def __set__(self, value):
#             self.rise_vel.water_density = value
# 
#     property water_viscosity:
#         def __get__(self):
#             return self.rise_vel.water_viscosity
# 
#         def __set__(self, value):
#             self.rise_vel.water_viscosity = value

#     def __repr__(self):
#         """
#         unambiguous repr of object, reuse for str() method
#         """
#         return "CyRiseVelocityMover(water_density=%s,water_viscosity=%s)" % \
#                 (self.water_density, self.water_viscosity)
# 
    def get_move(self,
                 model_time,
                 step_len,
                 cnp.ndarray[WorldPoint3D, ndim=1] ref_points,
                 cnp.ndarray[WorldPoint3D, ndim=1] delta,
                 cnp.ndarray[cnp.npy_double] rise_velocity,
                 cnp.ndarray[short] LE_status,
                 LEType spill_type):
        """
        .. function:: get_move(self,
                 model_time,
                 step_len,
                 np.ndarray[WorldPoint3D, ndim=1] ref_points,
                 np.ndarray[WorldPoint3D, ndim=1] delta,
                 cnp.ndarray[cnp.npy_double] rise_velocity,
                 np.ndarray[np.npy_int16] LE_status,
                 LE_type)

        Invokes the underlying C++ RiseVelocity_c.get_move(...)

        :param model_time: current model time
        :param step_len: step length over which delta is computed
        :param ref_points: current locations of LE particles
        :type ref_points: numpy array of WorldPoint3D
        :param delta: the change in position of each particle over step_len
        :type delta: numpy array of WorldPoint3D
        :param rise_velocity: rise_velocity to be applied to each particle
        :param le_status: status of each particle - movement is only
                          on particles in water
        :param spill_type: LEType defining whether spill is forecast
                           or uncertain
        :returns: none
        """
        cdef OSErr err
        N = len(ref_points)  # set a data type?

        err = self.rise_vel.get_move(N,
                                  model_time,
                                  step_len,
                                  &ref_points[0],
                                  &delta[0],
                                  &rise_velocity[0],
                                  &LE_status[0],
                                  spill_type,
                                  0)

        if err == 1:
            raise ValueError("Make sure ref_points, delta and rise_velocity"
                             " are defined")

        """
        Can probably raise this error before calling the C++ code,
        but the C++ also throwing this error
        """
        if err == 2:
            msg = "{0}: expected '{1}', got '{2}'"
            raise ValueError(msg.format('get_move()',
                                    "spill_type=('forecast','uncertainty')",
                                    spill_type))


def rise_velocity_from_drop_size(cnp.ndarray[cnp.npy_double, ndim=1] rise_vel,
                 cnp.ndarray[cnp.npy_double, ndim=1] le_density,
                 cnp.ndarray[cnp.npy_double] le_drop_size,
                 double water_viscosity,
                 double water_density):
    """
    """
    cdef OSErr vel_err
    vel_err = get_rise_velocity(len(rise_vel),
                                &rise_vel[0],
                                &le_density[0],
                                &le_drop_size[0],
                                water_viscosity,
                                water_density)

    if vel_err != 0:
        raise ValueError("C++ call to get_rise_velocity returned error code: "
                         "{0}".format(vel_err))
