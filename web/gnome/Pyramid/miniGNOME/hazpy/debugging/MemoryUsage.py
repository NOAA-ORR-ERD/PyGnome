#!/usr/bin/env python
"""MemoryUsage -- Estimate the amount of memory used by your Python program.

This module depends on Linux's /proc filesystem.

There is also a simpler method for estimating total memory, which used "ps" on Linux and mac
  GetMemory()


Usage:
mu = MemoryUsage()
mu.record("launch the program")
# Initialize some large data structures.
mu.record("do task 1")
# Initialize some more large data structures.
mu.record("do task 2")
# Print a report.
print "Maximum memory use was %0.2f megabytes." % mu.resident_mb()
print "Peak virtual memory use was %0.2f megabytes." % mu.vmpeak_mb()
print "Virtual memory size was %0.2f megabytes." % mu.vmsize_mb()
print "The resident samplings (in kilobytes) were", mu.readings_resident_kb()
print "The peak virtual samplings (in kilobytes) were", mu.readings_vmpeak_kb()
print "The virtual samplings (in kilobytes) were", mu.readings_vmsize_kb()

The meaning of the values derived for "vmsize" and "vmpeak" is not documented 
in "man proc".  I assume vmsize is the current amount of virtual memory used,
and vmpeak is the maximum amount that was used by this process, but this is
unverified.

This module assumes a megabyte is 1024 kilobytes, as is normal in computer
memory calculations.  This unit is more properly called a "mebibyte" (MiB),
see http://en.wikipedia.org/wiki/Megabyte .
"""
from __future__ import division
import re, os

resident_rx = re.compile( R"VmRSS:\s*(\d+) kB" )
vmpeak_rx   = re.compile( R"VmPeak:\s*(\d+) kB" )
vmsize_rx   = re.compile( R"VmSize:\s*(\d+) kB" )

class MemoryUsage(object):
    def __init__(self):
        self.readings_resident_kb  = []
        self.readings_vmpeak_kb = []
        self.readings_vmsize_kb = []
        self.labels = []

    def take_reading(self, label=None):
        f = open("/proc/self/status")   # Raises IOError.
        data = f.read()
        f.close()
        self._reading("resident", resident_rx, "VmRSS", data)
        self._reading("vmpeak",   vmpeak_rx,   "VmPeak", data)
        self._reading("vmsize",   vmsize_rx,   "VmSize", data)
        self.labels.append(label)
        
    def resident_mb(self):
        return self._result(self.readings_resident_kb)
        
    def vmpeak_mb(self):
        return self._result(self.readings_vmpeak_kb)
        
    def vmsize_mb(self):
        return self._result(self.readings_vmsize_kb)
        
    def summary(self):
        tup = self.resident_mb(), self.vmpeak_mb(), len(self.labels)
        msg = ("Memory used: %0.2f MB resident, %0.2f MB virtual "
               "(based on %d samplings).")
        return msg % tup

    #### PRIVATE INTERNAL METHODS ####
    def _reading(self, attr_base, rx, data_name, data):
        m = rx.search(data)
        if not m:
            tup = data_name, data
            m = "can't parse %s value in memory status, data follows:\n\n%s"
            raise OSError(m % tup)
        value = int(m.group(1))
        attr = "readings_%s_kb" % attr_base
        getattr(self, attr).append(value)
        
    def _result(self, readings):
        if not readings:
            return None
        return max(readings) / 1024

def GetMemory():
    """

    Returns the memory used by the current Python process

    """
    if os.name == "posix":
        ## this used to work on my Linux box:
        #memuse = float(os.popen("ps lwh "+`os.getpid()`).read().split()[6])
        ## this works on my Mac -- OS-X 10.4
        memuse = int(os.popen("ps lwh -p "+`os.getpid()`).readlines()[1].split()[6])
    else:
        print "GetMemory has not been implemented for a %s system yet"%(os.name)
        memuse = "???"
    return memuse

