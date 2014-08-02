"""
Declare the C++ mover classes from lib_gnome
"""
from libcpp cimport bool

from libc.stdint cimport int32_t

from type_defs cimport *
from utils cimport OSSMTimeValue_c  

from grids cimport GridVel_c, TimeGridVel_c
"""
Following are used by some of the methods of the movers
Currently, it looks like all methods are exposed from C++ so
these are declared here. This may get smaller as we work through the
cython files since there maybe no need to expose all the C++ functionality.
"""

'movers:'
cdef extern from "Mover_c.h":
    cdef cppclass Mover_c:
        OSErr PrepareForModelRun()
        OSErr PrepareForModelStep(Seconds &time, Seconds &time_step,
                                  bool uncertain, int numLESets,
                                  int32_t *LESetsSizesList)    # currently this happens in C++ get_move command
        void ModelStepIsDone()
        OSErr ReallocateUncertainty(int numLEs, short* LE_status)

cdef extern from "WindMover_c.h":
    cdef cppclass WindMover_c(Mover_c):
        WindMover_c() except +
        Boolean fIsConstantWind
        VelocityRec fConstantValue
        double fDuration
        double fUncertainStartTime
        double fSpeedScale
        double fAngleScale

        OSErr get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, double* windages, short* LE_status, LEType spillType, long spill_ID)
        void SetTimeDep(OSSMTimeValue_c *ossm)
        OSErr GetTimeValue(Seconds &time, VelocityRec *vel)

cdef extern from "Random_c.h":
    cdef cppclass Random_c(Mover_c):
        Random_c() except +
        double fDiffusionCoefficient
        double fUncertaintyFactor
        OSErr get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spillID)        

cdef extern from "RandomVertical_c.h":
    cdef cppclass RandomVertical_c(Mover_c):
        RandomVertical_c() except +
        double fVerticalDiffusionCoefficient
        double fVerticalBottomDiffusionCoefficient
        double fMixedLayerDepth
        OSErr get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spillID)        

cdef extern from "RiseVelocity_c.h":
    OSErr get_rise_velocity(int n, double *rise_vel, double *le_density, double *le_drop_size, double water_vis, double water_density)

    # the mover class, above is just a function for computing rise velocity
    cdef cppclass RiseVelocity_c(Mover_c):
        RiseVelocity_c() except +
        #double water_density
        #double water_viscosity
        OSErr get_move(int n, unsigned long model_time, unsigned long step_len,
                       WorldPoint3D* ref, WorldPoint3D* delta,
                       double* rise_velocity,
                       short* LE_status, LEType spillType, long spillID)        


cdef extern from "GridWindMover_c.h":
    cdef cppclass GridWindMover_c(WindMover_c):
        # Why can't I do this?
        #GridWindMover_c() except +
        TimeGridVel_c    *timeGrid
        Boolean fIsOptimizedForStep
        float    fWindScale
        float    fArrowScale
        short    fUserUnits

        GridWindMover_c ()
        WorldPoint3D    GetMove(Seconds&,Seconds&,Seconds&,Seconds&, long, long, LERec *, LETYPE)
        void 		    SetTimeGrid(TimeGridVel_c *newTimeGrid)
        OSErr           TextRead(char *path,char *topFilePath)
        OSErr          ExportTopology(char *topFilePath)
        void 		    SetExtrapolationInTime(bool extrapolate)
        bool 		    GetExtrapolationInTime()
        void 		    SetTimeShift(long timeShift)
        long 		    GetTimeShift()

cdef extern from "GridMap_c.h":
    cdef cppclass GridMap_c:
        GridVel_c    *fGrid
        WorldRect fMapBounds
        LONGH    fBoundarySegmentsH
        LONGH    fBoundaryTypeH
        LONGH    fBoundaryPointsH

        GridMap_c ()
        LONGH    GetBoundarySegs()
        LONGH    GetWaterBoundaries()
        LONGH    GetBoundaryPoints()
        OSErr 		    ExportTopology(char *path)
        OSErr 		    SaveAsNetCDF(char *path)
        OSErr           TextRead(char *path)

