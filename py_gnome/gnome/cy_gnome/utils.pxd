"""
lib_gnome utils
"""
from type_defs cimport *
from libcpp cimport bool
from libcpp.string cimport string
from libc.stdint cimport *

"""
MemUtils functions available from lib_gnome
"""
cdef extern from "MemUtils.h":
    Handle _NewHandle(long)
    void _DisposeHandleReally(Handle)
    long _GetHandleSize(Handle)

"""
Expose DateTime conversion functions from the lib_gnome/StringFunctions.h
"""
cdef extern from "StringFunctions.h":
    void DateToSeconds(DateTimeRec *, Seconds *)
    void SecondsToDate(Seconds, DateTimeRec *)

"""
Declare methods for interpolation of timeseries from 
lib_gnome/OSSMTimeValue_c class and ShioTimeValue
"""
cdef extern from "OSSMTimeValue_c.h":
    cdef cppclass OSSMTimeValue_c:
        OSSMTimeValue_c() except +
        # Members
        string          fileName
        string          filePath
        double          fScaleFactor
        WorldPoint3D    fStationPosition

        # make char array a string - easier to work with in Cython
        string          fStationName

        # Methods
        OSErr   GetTimeValue(Seconds &, VelocityRec *)
        OSErr   ReadTimeValues(char *, short, short)
        void    SetTimeValueHandle(TimeValuePairH)    # sets all time values
        TimeValuePairH GetTimeValueHandle()
        TimeValuePairH CalculateRunningAverage(long pastHoursToAverage, Seconds)
        short   GetUserUnits()
        void    SetUserUnits(short)
        OSErr   CheckStartTime(Seconds)
        void    Dispose()
        WorldPoint3D    GetStationLocation()

"""
ShioTimeValue_c.h derives from OSSMTimeValue_c - so no need to redefine methods
given in OSSMTimeValue_c
"""
cdef extern from "ShioTimeValue_c.h":
    ctypedef struct EbbFloodData:
        Seconds time
        double speedInKnots
        short type  # // 0 -> MinBeforeFlood, 1 -> MaxFlood, 2 -> MinBeforeEbb, 3 -> MaxEbb

    ctypedef EbbFloodData *EbbFloodDataP
    ctypedef EbbFloodData **EbbFloodDataH  # Weird syntax, it says EbbFloodDataH is pointer to pointer to EbbFloodData struct

    ctypedef struct HighLowData:
        Seconds time
        double height
        short type  # // 0 -> Low Tide, 1 -> High Tide

    ctypedef HighLowData *HighLowDataP
    ctypedef HighLowData **HighLowDataH

    cdef cppclass ShioTimeValue_c(OSSMTimeValue_c):
        ShioTimeValue_c() except +
        char        fStationType
        string      fYearDataPath
        bool        daylight_savings_off    # is this required?
        EbbFloodDataH   fEbbFloodDataHdl    # values to show on list for tidal currents - not sure if these should be available
        HighLowDataH    fHighLowDataHdl

        OSErr       ReadTimeValues(char *path)
        OSErr       SetYearDataPath(char *path)

        # Not Sure if Following are required/used
        OSErr       GetConvertedHeightValue(Seconds, VelocityRec *)
        OSErr       GetProgressiveWaveValue(Seconds &, VelocityRec *)

cdef extern from "Weatherers_c.h":
    # OSErr emulsify(int n, unsigned long step_len, double *frac_water, double *interfacial_area, double *frac_evap, double *droplet_diameter, unsigned long *age, unsigned long *bulltime, double k_emul, unsigned long emul_time, double emul_C, double S_max, double Y_max, double drop_max)
    OSErr emulsify(int n, unsigned long step_len, double *frac_water, double *interfacial_area, double *frac_evap, int32_t *age, double *bulltime, double k_emul, double emul_time, double emul_C, double S_max, double Y_max, double drop_max)
    OSErr adios2_disperse(int n, unsigned long step_len, double *frac_water, double *le_mass, double *le_viscosity, double *le_density, double *fay_area, double *d_disp, double *d_sed, double frac_breaking_waves, double disp_wave_energy, double wave_height, double visc_w, double rho_w, double C_sed, double V_entrain, double ka)
