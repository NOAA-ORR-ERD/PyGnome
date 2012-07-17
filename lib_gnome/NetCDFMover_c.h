/*
 *  NetCDFMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __NetCDFMover_c__
#define __NetCDFMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "CurrentMover_c.h"

#ifndef pyGNOME
#include "GridVel.h"
#else
#include "GridVel_c.h"
#define TGridVel GridVel_c
#define TMap Map_c
#endif

typedef struct {
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
} NetCDFVariables;

enum { REGULAR=1, REGULAR_SWAFS, CURVILINEAR, TRIANGULAR, REGRIDDED};	// maybe eliminate regridded option

enum {
	I_NETCDFNAME = 0 ,
	I_NETCDFACTIVE, 
	I_NETCDFGRID, 
	I_NETCDFARROWS,
	I_NETCDFSCALE,
	I_NETCDFUNCERTAINTY,
	I_NETCDFSTARTTIME,
	I_NETCDFDURATION, 
	I_NETCDFALONGCUR,
	I_NETCDFCROSSCUR,
	//I_NETCDFMINCURRENT
};

class NetCDFMover_c : virtual public CurrentMover_c {

public:
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
	//double fOffset_u;
	//double fOffset_v;
	//double fCurScale_u;
	//double fCurScale_v;

#ifndef pyGNOME
	NetCDFMover_c (TMap *owner, char *name);
#endif
	NetCDFMover_c () {fTimeHdl = 0; fGrid = 0; fDepthLevelsHdl = 0; fDepthLevelsHdl2 = 0; fDepthsH = 0; fDepthDataInfo = 0; fInputFilesHdl = 0; fLESetSizesH = 0; fUncertaintyListH = 0;} // AH 07/17/2012
	
	virtual ClassID 	GetClassID () { return TYPE_NETCDFMOVER; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_NETCDFMOVER) return TRUE; return CurrentMover_c::IAm(id); }

	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	virtual long 		GetVelocityIndex(WorldPoint p);
	virtual LongPoint 	GetVelocityIndices(WorldPoint wp);
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(const Seconds& start_time, const Seconds& stop_time, const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	void 				GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2);
	float 				GetMaxDepth();
	virtual float		GetArrowDepth() {return fVar.arrowDepth;}
	
	virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	
	long 					GetNumDepths(void);
	virtual long 		GetNumDepthLevels();
	Seconds 				GetTimeValue(long index);
	virtual OSErr		GetStartTime(Seconds *startTime);
	virtual OSErr		GetEndTime(Seconds *endTime);
	virtual double 	GetStartUVelocity(long index);
	virtual double 	GetStartVVelocity(long index);
	virtual double 	GetEndUVelocity(long index);
	virtual double 	GetEndVVelocity(long index);
	virtual double	GetDepthAtIndex(long depthIndex, double totalDepth);
#ifndef pyGNOME
	virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
#endif
	float		GetTotalDepth(WorldPoint refPoint, long triNum);
	virtual WorldPoint3D       GetMove(const Seconds& start_time, const Seconds& stop_time, const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, const Seconds&, const Seconds&, bool); // AH 07/10/2012
	virtual void 		ModelStepIsDone();
	
	long 					GetNumTimesInFile();
	long 					GetNumFiles();
	virtual OSErr 		CheckAndScanFile(char *errmsg, const Seconds& start_time, const Seconds& model_time);	// AH 07/17/2012
	virtual OSErr	 	SetInterval(char *errmsg, const Seconds& start_time, const Seconds& model_time);	// AH 07/17/2012
	OSErr 				ScanFileForTimes(char *path,Seconds ***timeH,Boolean setStartTime, const Seconds& start_time);	// AH 07/17/2012
	
	virtual Boolean 	CheckInterval(long &timeDataInterval, const Seconds& start_time, const Seconds& model_time);	// AH 07/17/2012
	virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
	void 				DisposeLoadedData(LoadedData * dataPtr);	
	void 				ClearLoadedData(LoadedData * dataPtr);
	void 				DisposeAllLoadedData();
	
	virtual long 		GetNumDepthLevelsInFile();	// eventually get rid of this
	//virtual DepthValuesSetH 	GetDepthProfileAtPoint(WorldPoint refPoint) {return nil;}
	virtual OSErr 	GetDepthProfileAtPoint(WorldPoint refPoint, long timeIndex, DepthValuesSetH *profilesH) {*profilesH=nil; return 0;}
	


};

#undef TMap
#endif
