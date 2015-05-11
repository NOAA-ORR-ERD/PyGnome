'''
Wrappers around C++ lib_gnome current_movers
'''

from libcpp cimport bool

from libc.stdint cimport int32_t

from type_defs cimport *
from utils cimport OSSMTimeValue_c

from movers cimport Mover_c
from grids cimport TimeGridVel_c
from grids cimport GridVel_c

cdef extern from "CurrentMover_c.h":
    cdef cppclass CurrentMover_c(Mover_c):
        double fDuration
        double fUncertainStartTime
        double fUpCurUncertainty
        double fDownCurUncertainty
        double fRightCurUncertainty
        double fLeftCurUncertainty

cdef extern from "CATSMover_c.h":
    cdef cppclass CATSMover_c(CurrentMover_c):
        CATSMover_c() except +
        GridVel_c		*fGrid
        double          fEddyDiffusion
        double          fEddyV0
        short           scaleType   # set this automatically!
        double          scaleValue
        double          refScale
        Boolean         bTimeFileActive

        int             TextRead(char* path)
        void            SetRefPosition(WorldPoint3D p)
        WorldPoint3D    GetRefPosition()
        OSErr    InitMover()

        OSErr get_move(int n, unsigned long model_time, unsigned long step_len,
                       WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status,
                       LEType spillType, long spillID)
        void  SetTimeDep(OSSMTimeValue_c *ossm)
        LongPointHdl  GetPointsHdl()
        WORLDPOINTH  GetWorldPointsHdl()
        VelocityFH  GetVelocityHdl()
        TopologyHdl  GetTopologyHdl()
        WORLDPOINTH  GetTriangleCenters()


cdef extern from "ComponentMover_c.h":
    cdef cppclass ComponentMover_c(CurrentMover_c):
        ComponentMover_c() except +
        double          pat1Angle
        double          pat2Angle
        double          pat1Speed
        double          pat2Speed
        long            pat1SpeedUnits
        long            pat2SpeedUnits
        double          pat1ScaleToValue
        double          pat2ScaleToValue
        Boolean         bUseAveragedWinds
        Boolean         bExtrapolateWinds
        double          fScaleFactorAveragedWinds
        double          fPowerFactorAveragedWinds
        long            fPastHoursToAverage
        long            scaleBy

        int             TextRead(char* catsPath1, char* catsPath2)
        void            SetRefPosition(WorldPoint3D p)
        WorldPoint3D    GetRefPosition()

        OSErr get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spillID)
        void  SetTimeFile(OSSMTimeValue_c *ossm)    


cdef extern from "GridCurrentMover_c.h":

    cdef struct UncertaintyParameters:
        double     alongCurUncertainty
        double     crossCurUncertainty
        double     uncertMinimumInMPS
        double     startTimeInHrs
        double     durationInHrs

    cdef cppclass GridCurrentMover_c(CurrentMover_c):
        UncertaintyParameters fUncertainParams
        double fCurScale
        TimeGridVel_c    *timeGrid
        Boolean fIsOptimizedForStep
        Boolean fAllowVerticalExtrapolationOfCurrents

        GridCurrentMover_c ()
        WorldPoint3D    GetMove(Seconds&,Seconds&,Seconds&,Seconds&, long, long, LERec *, LETYPE)
        OSErr             get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spillID)
        void             SetTimeGrid(TimeGridVel_c *newTimeGrid)
        OSErr           TextRead(char *path,char *topFilePath)
        OSErr           ExportTopology(char *topFilePath)
        void             SetExtrapolationInTime(bool extrapolate)
        bool             GetExtrapolationInTime()
        void             SetTimeShift(long timeShift)
        long             GetTimeShift()

cdef extern from "CurrentCycleMover_c.h":

    cdef cppclass CurrentCycleMover_c(GridCurrentMover_c):
        OSSMTimeValue_c *timeDep
        Boolean bTimeFileActive
        short fPatternStartPoint
        WorldPoint      refP

        CurrentCycleMover_c ()
        WorldPoint3D    GetMove(Seconds&,Seconds&,Seconds&,Seconds&, long, long, LERec *, LETYPE)
        #OSErr             get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spillID)
        #void             SetTimeGrid(TimeGridVel_c *newTimeGrid)
        #OSErr           TextRead(char *path,char *topFilePath)
        #OSErr           ExportTopology(char *topFilePath)
        void              SetTimeDep(OSSMTimeValue_c *ossm)
        void              SetRefPosition(WorldPoint3D p)
        WorldPoint3D      GetRefPosition()