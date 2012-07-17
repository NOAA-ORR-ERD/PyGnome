include "type_defs.pxi"
include "current_mover.pxi"
include "grid_vel.pxi"

cdef extern from "NetCDFMover_c.h":
    
    cdef struct NetCDFVariables:
        char		*pathName
        char		*userName
        double 	alongCurUncertainty
        double 	crossCurUncertainty
        double 	uncertMinimumInMPS
        double 	curScale
        double 	startTimeInHrs
        double 	durationInHrs
        long		maxNumDepths
        short		gridType
        Boolean 	bShowGrid
        Boolean 	bShowArrows
        Boolean	bUncertaintyPointOpen
        double 	arrowScale
        double 	arrowDepth

    cdef cppclass NetCDFMover_c(CurrentMover_c):
        long fNumRows
        long fNumCols
        long fNumDepthLevels
        NetCDFVariables fVar
        Boolean bShowDepthContours
        Boolean bShowDepthContourLabels
        GridVel_c	*fGrid
        Seconds **fTimeHdl
        float **fDepthLevelsHdl
        float **fDepthLevelsHdl2
        float hc
        float **fDepthsH
        float fFillValue
        double fFileScaleFactor
        Boolean fIsNavy
        Boolean fIsOptimizedForStep
        Boolean fOverLap
        Seconds fOverLapStartTime
        long fTimeShift
        Boolean fAllowExtrapolationOfCurrentsInTime
        Boolean fAllowVerticalExtrapolationOfCurrents
        float	fMaxDepthForExtrapolation
        
        NetCDFMover_c ()
        WorldPoint3D        GetMove(Seconds&,Seconds&,Seconds&,Seconds&, long, long, LERec *, LETYPE)
#        OSErr 		        ReadTimeData(long index,VelocityFH *velocityH, char* errmsg)
#        void 				DisposeLoadedData(LoadedData * dataPtr)
#        void 				ClearLoadedData(LoadedData * dataPtr)
#        void 				DisposeAllLoadedData()
        
        
