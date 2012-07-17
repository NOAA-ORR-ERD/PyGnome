include "type_defs.pxi"
include "current_mover.pxi"
include "grid_vel.pxi"

cdef extern from "NetCDFMover_c.h":
    
    cdef struct NetCDFVariables:
        char		pathName[kMaxNameLen];
        char		userName[kPtCurUserNameLen]; // user name for the file, or short file name
        //char		userName[kMaxNameLen]; // user name for the file, or short file name - might want to allow longer names...
        double 	alongCurUncertainty;	
        double 	crossCurUncertainty;	
        double 	uncertMinimumInMPS;	
        double 	curScale;	// use value from file? 	
        double 	startTimeInHrs;	
        double 	durationInHrs;	
        //
        //long		numLandPts; // 0 if boundary velocities defined, else set boundary velocity to zero
        long		maxNumDepths;
        short		gridType;
        //double	bLayerThickness;
        //
        Boolean 	bShowGrid;
        Boolean 	bShowArrows;
        Boolean	bUncertaintyPointOpen;
        double 	arrowScale;
        double 	arrowDepth;	// depth level where velocities will be shown

    cdef cppclass NetCDFMover_c(CurrentMover_c):
        long fNumRows;
        long fNumCols;
        long fNumDepthLevels;
        NetCDFVariables fVar;
        Boolean bShowDepthContours;
        Boolean bShowDepthContourLabels;
        TGridVel	*fGrid;	//VelocityH		grid; 
        //PtCurTimeDataHdl fTimeDataHdl;
        Seconds **fTimeHdl;
        float **fDepthLevelsHdl;	// can be depth levels, sigma, or sc_r (for roms formula)
        float **fDepthLevelsHdl2;	// Cs_r (for roms formula)
        float hc;	// parameter for roms formula
        LoadedData fStartData; 
        LoadedData fEndData;
        FLOATH fDepthsH;	// check what this is, maybe rename
        DepthDataInfoH fDepthDataInfo;
        float fFillValue;
        double fFileScaleFactor;
        Boolean fIsNavy;	// special variable names for Navy, maybe change to grid type depending on Navy options
        Boolean fIsOptimizedForStep;
        Boolean fOverLap;
        Seconds fOverLapStartTime;
        PtCurFileInfoH	fInputFilesHdl;
        //Seconds fTimeShift;		// to convert GMT to local time
        long fTimeShift;		// to convert GMT to local time
        Boolean fAllowExtrapolationOfCurrentsInTime;
        Boolean fAllowVerticalExtrapolationOfCurrents;
        float	fMaxDepthForExtrapolation;
        Rect fLegendRect;

        NetCDFMover_c ()
        
        virtual WorldPoint3D       GetMove(const Seconds& start_time, const Seconds& stop_time, const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
        
        virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
        void 				DisposeLoadedData(LoadedData * dataPtr);	
        void 				ClearLoadedData(LoadedData * dataPtr);
        void 				DisposeAllLoadedData();
        
        
