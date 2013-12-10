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
#include "ExportSymbols.h"
//#include "TimeGridVel_c.h"

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
	double 	alongCurUncertainty;	
	double 	crossCurUncertainty;	
	double 	uncertMinimumInMPS;	
	double 	curScale;	// use value from file? 	
	double 	startTimeInHrs;	
	double 	durationInHrs;	
	//
	long		maxNumDepths;
	short		gridType;
	//
	Boolean 	bShowGrid;
	Boolean 	bShowArrows;
	Boolean	bUncertaintyPointOpen;
	double 	arrowScale;
	double 	arrowDepth;	// depth level where velocities will be shown
} NetCDFVariables;

//enum { REGULAR=1, REGULAR_SWAFS, CURVILINEAR, TRIANGULAR, REGRIDDED};	// maybe eliminate regridded option

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
};

//class DLL_API NetCDFMover_c : virtual public CurrentMover_c {
class NetCDFMover_c : virtual public CurrentMover_c {
// not using this class for pyGNOME
public:
	long fNumRows;
	long fNumCols;
	long fNumDepthLevels;
	NetCDFVariables fVar;
	Boolean bShowDepthContours;
	Boolean bShowDepthContourLabels;
	TGridVel	*fGrid;	//VelocityH		grid; 
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
	long fTimeShift;		// to convert GMT to local time
	Boolean fAllowExtrapolationOfCurrentsInTime;
	Boolean fAllowVerticalExtrapolationOfCurrents;
	float	fMaxDepthForExtrapolation;
	Rect fLegendRect;
	//TimeGridVel_c *netcdfGrid;

#ifndef pyGNOME
	NetCDFMover_c (TMap *owner, char *name);
#endif
	NetCDFMover_c () {fTimeHdl = 0; fGrid = 0; fDepthLevelsHdl = 0; fDepthLevelsHdl2 = 0; fDepthsH = 0; fDepthDataInfo = 0; fInputFilesHdl = 0; fLESetSizesH = 0; fUncertaintyListH = 0;} // AH 07/17/2012
	
	//virtual ClassID 	GetClassID () { return TYPE_NETCDFMOVER; }
	//virtual Boolean	IAm(ClassID id) { if(id==TYPE_NETCDFMOVER) return TRUE; return TCurrentMover::IAm(id); }

	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	virtual long 		GetVelocityIndex(WorldPoint p);
	virtual LongPoint 	GetVelocityIndices(WorldPoint wp);
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
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
	virtual OSErr 		PrepareForModelRun(); 
	float		GetTotalDepth(WorldPoint refPoint, long triNum);
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList); 
	virtual void 		ModelStepIsDone();
	
	long 					GetNumTimesInFile();
	long 					GetNumFiles();
	virtual OSErr 		CheckAndScanFile(char *errmsg, const Seconds& model_time);	
	virtual OSErr	 	SetInterval(char *errmsg, const Seconds& model_time);	
	OSErr 				ScanFileForTimes(char *path,Seconds ***timeH,Boolean setStartTime);	
	
	virtual Boolean 	CheckInterval(long &timeDataInterval, const Seconds& model_time);	
	virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
	void 				DisposeLoadedData(LoadedData * dataPtr);	
	void 				ClearLoadedData(LoadedData * dataPtr);
	void 				DisposeAllLoadedData();
	
	virtual long 		GetNumDepthLevelsInFile();	// eventually get rid of this
	virtual OSErr 	GetDepthProfileAtPoint(WorldPoint refPoint, long timeIndex, DepthValuesSetH *profilesH) {*profilesH=nil; return 0;}
	

			OSErr		get_move(int n, Seconds model_time, Seconds step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spill_ID);

};

#undef TMap
#endif
