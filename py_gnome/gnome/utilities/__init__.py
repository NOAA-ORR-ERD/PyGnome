"""
__init__.py for the utilities package

a few small things here, 'cause why not?

"""

import math
import operator
import sys
import warnings

import numpy as np

#from .sig_fig_rounding import RoundToSigFigs_fp as round_sf

div = {'GB': 1024*1024*1024,
       'MB': 1024*1024,
       'KB': 1024,
       }

def convert_longitude(lon, coord_system='-180--180'):
    """
    Convert longitude values to a given coordinate system.

    Options are:

    "-180--180": Negative 180 degrees to 180 degrees

    "0--360": Zero to 360 degrees

    :param lon: numpy array-like of longitude values of float type
    :param  coord_system='-180--180': options are: {"-180--180", "0--360"}

    NOTE: this function also normalizes so that:

    360 converts to 0
    -180 converts to 180

    It should be safe to call this on any coords -- if they are already
    in the expected format, they will not be changes, except for the
    normalization above.
    """

    if coord_system not in {"-180--180", "0--360"}:
        raise TypeError('coord_system must be one of {"-180--180", "0--360"}')
    lon = np.array(lon)

    if coord_system == "0--360":
        return (lon + 360) % 360
    elif coord_system == "-180--180":
        lon[lon > 180] -= 360
        lon[lon <= -180] += 360
        return lon


# getting memory usage.
if sys.platform.startswith('win'):
    """
    Functions for getting memory usage of Windows processes.

    from:

    http://code.activestate.com/recipes/578513-get-memory-usage-of-windows-processes-using-getpro/

    get_mem_use(units='MB') is the one to get memory use
    for the current process.
    """
    import ctypes
    from ctypes import wintypes

    GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
    GetCurrentProcess.argtypes = []
    GetCurrentProcess.restype = wintypes.HANDLE

    SIZE_T = ctypes.c_size_t

    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ('cb', wintypes.DWORD),
            ('PageFaultCount', wintypes.DWORD),
            ('PeakWorkingSetSize', SIZE_T),
            ('WorkingSetSize', SIZE_T),
            ('QuotaPeakPagedPoolUsage', SIZE_T),
            ('QuotaPagedPoolUsage', SIZE_T),
            ('QuotaPeakNonPagedPoolUsage', SIZE_T),
            ('QuotaNonPagedPoolUsage', SIZE_T),
            ('PagefileUsage', SIZE_T),
            ('PeakPagefileUsage', SIZE_T),
            ('PrivateUsage', SIZE_T),
        ]

    GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
    GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX),
        wintypes.DWORD,
    ]
    GetProcessMemoryInfo.restype = wintypes.BOOL

    def get_current_process():
        """Return handle to current process."""
        return GetCurrentProcess()

    def get_memory_info(process=None):
        """Return Win32 process memory counters structure as a dict."""
        if process is None:
            process = get_current_process()
        counters = PROCESS_MEMORY_COUNTERS_EX()
        ret = GetProcessMemoryInfo(process, ctypes.byref(counters),
                                   ctypes.sizeof(counters))
        if not ret:
            raise ctypes.WinError()
        info = dict((name, getattr(counters, name))
                    for name, _ in counters._fields_)
        return info

    def get_mem_use(units='MB'):
        """
        returns the total memory use of the current python process

        :param units='MB': the units you want the reslut in. Options are:
                           'GB', 'MB', 'KB'
        """
        info = get_memory_info()
        return info['PrivateUsage'] / float(div[units])


else:  # for posix systems only tested on OS-X for now
    def get_mem_use(units='MB'):
        """
        returns the total memory use of the current python process

        :param units='MB': the units you want the reslut in. Options are:
                           'GB', 'MB', 'KB'
        """
        import resource
        useage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        div = {'GB': 1024*1024*1024,
               'MB': 1024*1024,
               'KB': 1024,
               }
        d = div[units]
        if sys.platform == 'darwin':
            pass
        elif sys.platform.startswith("linux"):
            d /= 1024
        else:
            warnings.warn('memory use reported may not be correct '
                          'on this platform')
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / float(d)


# NOTE: there is a theoretically robust proper numpy impimentation here:
#       https://github.com/odysseus9672/SELPythonLibs/blob/master/SigFigRounding.py
#       but I found it didn't quite work for some values :-(, for example:
# In [18]: x
# Out[18]: 3.3456789e-20

# In [19]: sig
# Out[19]: 4

# In [20]: RoundToSigFigs_fp(x, sig)
# Out[20]: 3.3459999999999996e-20

def _round_sf_float(x, sigfigs):
    """
    round a float to significant figures -- no error checking

    This uses the "g" format specifier -- maybe slow, but robust for
    the purpose of getting the display what we want.
    """
    # doing it with math -- mostly worked, but got an odd issue when passing through arrays
    # if x == 0:
    #     return 0.0
    # if math.isnan(x):
    #     return math.nan
    # elif math.isinf(x):
    #     return x
    # else:
    #     result = float(f"{x:}")round(x, sigfigs - int(math.floor(math.log10(abs(x)))) - 1)

    formatter = "{x:.%d}" % sigfigs  # using "old style formatting for the curly bracket"
    result = float(formatter.format(x=x))
    return result


def round_sf_scalar(x, sigfigs):
    x = float(x)
    sigfigs = operator.index(sigfigs)
    return _round_sf_float(x, sigfigs)


def round_sf_array(x, sigfigs):
    """
    round to a certain number of significant figures

    should work on floats and numpy arrays

    NOTE: This should be vectorizable, but np.round takes only a scalar value
          for number of decimals -- you could vectorize the rest of the computation,
          and then loop for the round, but would that buy anything?
          (or use np.vectorize)
    """
    sigfigs = operator.index(sigfigs)
    x = np.asarray(x, dtype=np.float64)
    x = np.asarray(x)

    shape = x.shape
    result = np.fromiter((_round_sf_float(val, sigfigs) for val in x.flat), dtype=np.float64)
    result.shape = shape
    return result


