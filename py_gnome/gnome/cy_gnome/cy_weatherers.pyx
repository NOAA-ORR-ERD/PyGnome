cimport numpy as cnp
import numpy as np

# following exist in gnome.cy_gnome
from type_defs cimport *
from utils cimport emulsify


def emulsify_oil(step_len, cnp.ndarray[cnp.npy_double] frac_water,
                 cnp.ndarray[cnp.npy_double] le_interfacial_area,
                 cnp.ndarray[cnp.npy_double] le_frac_evap,
                 cnp.ndarray[cnp.npy_double] le_droplet_diameter,
                 cnp.ndarray[unsigned long] le_age,
                 #cnp.ndarray[unsigned long] le_bulltime,
                 cnp.ndarray[cnp.npy_double] le_bulltime,
                 double k_emul,
                 #emul_time,
                 double emul_time,
                 double emul_C,
                 double S_max,
                 double Y_max,
                 double drop_max):
    """
    """
    cdef OSErr vel_err
    N = len(frac_water)  
        
    vel_err = emulsify(N,
                                step_len,
                                &frac_water[0],
                                &le_interfacial_area[0],
                                &le_frac_evap[0],
                                &le_droplet_diameter[0],
                                &le_age[0],
                                &le_bulltime[0],
                                k_emul,
                                emul_time,
                                emul_C,
                                S_max,
                                Y_max,
                                drop_max)

    if vel_err != 0:
        raise ValueError("C++ call to emulsify returned error code: "
                         "{0}".format(vel_err))
