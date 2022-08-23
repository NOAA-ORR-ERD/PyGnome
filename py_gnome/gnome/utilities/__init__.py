"""
__init__.py for the utilities package

a few small things here, 'cause why not?

"""

import sys
import warnings

import numpy as np

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
