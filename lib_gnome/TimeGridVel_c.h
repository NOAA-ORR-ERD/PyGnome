/*
 *  TimeGridVel_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __TimeGridVel_c__
#define __TimeGridVel_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "ExportSymbols.h"
#include "DagTree.h"
#include "DagTreeIO.h"
#include "my_build_list.h"

#ifndef pyGNOME
#include "GridVel.h"
#else
#include "GridVel_c.h"
#define TGridVel GridVel_c
#endif

// code goes here, decide which fields go with the mover
typedef struct {
	char		pathName[kMaxNameLen];
	char		userName[kPtCurUserNameLen]; // user name for the file, or short file name
	double 	fileScaleFactor;	// use value from file? 	
	//
	long		maxNumDepths;
	short		gridType;
	//
} TimeGridVariables;

Boolean IsNetCDFFile (char *path, short *gridType);
Boolean IsNetCDFPathsFile (char *path, Boolean *isNetCDFPathsFile, char *fileNamesPath, short *gridType);
Boolean IsPtCurFile (char *path);
Boolean IsGridCurTimeFile (char *path, short *selectedUnits);
Boolean IsGridWindFile(char *path,short *selectedUnits);
class TimeGridVel_c
{
public:
	
	long fNumRows;
	long fNumCols;
	TimeGridVariables fVar;
	TGridVel	*fGrid;	//VelocityH		grid; 
	Seconds **fTimeHdl;
	LoadedData fStartData; 
	LoadedData fEndData;

	float fFillValue;
	//double fFileScaleFactor;	
	Boolean fOverLap;
	Seconds fOverLapStartTime;
	PtCurFileInfoH	fInputFilesHdl;
	long fTimeShift;		// to convert GMT to local time
	Boolean fAllowExtrapolationInTime;
	
	WorldRect fGridBounds;


	TimeGridVel_c (/*TMover *owner, char *name*/);	// do we need an owner? or a name

	//virtual	~TimeGridVel_c() { Dispose (); }
	
	//virtual void	Dispose() { return; }
	
	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDVEL; }
	virtual Boolean	IAm(ClassID id) { return(id==TYPE_TIMEGRIDVEL);}

	virtual long 		GetVelocityIndex(WorldPoint p);
	virtual LongPoint 	GetVelocityIndices(WorldPoint wp);
	virtual VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint3D p) {VelocityRec vRec = {0,.0,}; return vRec;}
	
	//virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	
	//virtual void SetGridBounds(WorldRect gridBounds){return fGrid->SetBounds(gridBounds);}	
	virtual WorldRect GetGridBounds(){return fGridBounds;}	
	virtual void SetGridBounds(WorldRect gridBounds){fGridBounds = gridBounds;}	
	virtual Seconds 		GetStartTimeValue(long index);
	virtual Seconds 		GetTimeValue(long index);
	virtual OSErr		GetStartTime(Seconds *startTime);
	virtual OSErr		GetEndTime(Seconds *endTime);
	virtual double 	GetStartUVelocity(long index);
	virtual double 	GetStartVVelocity(long index);
	virtual double 	GetEndUVelocity(long index);
	virtual double 	GetEndVVelocity(long index);
	
	long 					GetNumTimesInFile();
	long 					GetNumFiles();
	virtual OSErr 		CheckAndScanFile(char *errmsg, const Seconds& model_time);	
	virtual OSErr	 	SetInterval(char *errmsg, const Seconds& model_time);	
	//virtual OSErr 		ScanFileForTimes(char *path,Seconds ***timeH,Boolean setStartTime);	
	
	virtual Boolean 	CheckInterval(long &timeDataInterval, const Seconds& model_time);	
	virtual OSErr		TextRead(char *path,char *topFilePath) {return 0;}
	virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg) {return 0;}
	OSErr 				ReadInputFileNames(char *fileNamesPath);
	//void				SetInputFilesHdl(PtCurFileInfoH inputFilesHdl) {if (fInputFilesHdl) {DisposeHandle((Handle)fInputFilesHdl)} fInputFilesHdl = inputFilesHdl;}

	virtual void		DisposeTimeHdl();
	void 				DisposeLoadedData(LoadedData * dataPtr);	
	void 				ClearLoadedData(LoadedData * dataPtr);
	void 				DisposeAllLoadedData();
};
OSErr 		ScanFileForTimes(char *path,Seconds ***timeH);	
OSErr 		ScanFileForTimes(char *path,PtCurTimeDataHdl ***timeDataH,Seconds ***timeH);	

class TimeGridVelRect_c : virtual public TimeGridVel_c
{
public:
	
	long fNumDepthLevels;
	float **fDepthLevelsHdl;	// can be depth levels, sigma, or sc_r (for roms formula)
	float **fDepthLevelsHdl2;	// Cs_r (for roms formula)
	float hc;	// parameter for roms formula

	FLOATH fDepthsH;	// check what this is, maybe rename
	DepthDataInfoH fDepthDataInfo;
	double fFileScaleFactor;
	Boolean fIsNavy;	// special variable names for Navy, maybe change to grid type depending on Navy options	Boolean fIsOptimizedForStep;

	Boolean fAllowVerticalExtrapolationOfCurrents;
	float	fMaxDepthForExtrapolation;
	
	
	TimeGridVelRect_c();
	//virtual	~TimeGridVelRect_c() { Dispose (); }
	
	//virtual void	Dispose() { return; }
	
	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDVELRECT; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEGRIDVELRECT) return TRUE; return TimeGridVel_c::IAm(id); }
	
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint3D p);
	void 				GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2);
	float 				GetMaxDepth();
	
	long 					GetNumDepths(void);
	virtual long 		GetNumDepthLevels();

	virtual double	GetDepthAtIndex(long depthIndex, double totalDepth);
	float		GetTotalDepth(WorldPoint refPoint, long triNum);
	virtual OSErr		TextRead(char *path,char *topFilePath);
	virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg);
	
	virtual long 		GetNumDepthLevelsInFile();	// eventually get rid of this
};

class TimeGridVelCurv_c : virtual public TimeGridVelRect_c
{
public:
	
	
	LONGH fVerdatToNetCDFH;	// for curvilinear
	WORLDPOINTFH fVertexPtsH;		// for curvilinear, all vertex points from file
	Boolean bVelocitiesOnNodes;		// default is velocities on cells

	//virtual	~TimeGridVelCurv_c() { Dispose (); }
	
	//virtual void	Dispose() { return; }
	
	TimeGridVelCurv_c ();
	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDVELCURV; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEGRIDVELCURV) return TRUE; return TimeGridVelRect_c::IAm(id); }
	
	LongPointHdl		GetPointsHdl();
	virtual OSErr		TextRead(char *path,char *topFilePath);
	OSErr 				ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
	VelocityRec			GetScaledPatValue(const Seconds& model_time, WorldPoint3D refPoint);

	OSErr 				ReorderPoints(DOUBLEH landmaskH, char* errmsg); 
	OSErr 				ReorderPointsNoMask(char* errmsg); 
	OSErr 				ReorderPointsCOOPSMask(DOUBLEH landmaskH, char* errmsg); 
	Boolean				IsCOOPSFile();
	
	virtual long 		GetVelocityIndex(WorldPoint wp);
	virtual LongPoint 	GetVelocityIndices(WorldPoint wp);
	OSErr 				GetLatLonFromIndex(long iIndex, long jIndex, WorldPoint *wp);
	virtual long 		GetNumDepthLevels();
	void 				GetDepthIndices(long ptIndex, float depthAtPoint, float totalDepth, long *depthIndex1, long *depthIndex2);
	float		GetTotalDepthFromTriIndex(long triIndex);
	float		GetTotalDepth(WorldPoint refPoint,long ptIndex);
	
	virtual	OSErr 	ReadTopology(char* path);
	//virtual	OSErr 	ReadTopology(char* path, TMap **newMap);
	//virtual	OSErr 	ExportTopology(char* path);
};

class TimeGridVelTri_c : virtual public TimeGridVelCurv_c
{
public:
	
	long fNumNodes;
	long fNumEles;	//for now, number of triangles
	Boolean bVelocitiesOnTriangles;
	
	
	//virtual	~TimeGridVelTri_c() { Dispose (); }
	
	//virtual void	Dispose() { return; }
	
	TimeGridVelTri_c ();
	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDVELTRI; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEGRIDVELTRI) return TRUE; return TimeGridVelCurv_c::IAm(id); }
	LongPointHdl			GetPointsHdl();
	void					GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2);
	virtual OSErr		TextRead(char *path,char *topFilePath);
	OSErr 				ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint3D refPoint);
	VelocityRec 		GetScaledPatValue3D(const Seconds& model_time, InterpolationVal interpolationVal,float depth);
	OSErr					ReorderPoints(long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts); 
	OSErr					ReorderPoints2(long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts, long *tri_verts, long *tri_neighbors, long ntri, Boolean isCCW);
	
	virtual long			GetNumDepthLevels();
	float					GetTotalDepth(WorldPoint refPoint, long triNum);
	
	virtual	OSErr 	ReadTopology(char* path);
	//virtual	OSErr 	ReadTopology(char* path, TMap **newMap);
	//virtual	OSErr 	ExportTopology(char* path);
};

//#ifndef pyGNOME
class TimeGridCurRect_c : virtual public TimeGridVel_c
{
public:
	
	PtCurTimeDataHdl fTimeDataHdl;	
	short fUserUnits;
	
	TimeGridCurRect_c();
	//virtual	~TimeGridCurRect_c() { Dispose (); }
	
	//virtual void	Dispose() { return; }
	
	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDCURRECT; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEGRIDCURRECT) return TRUE; return TimeGridVel_c::IAm(id); }
	
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint3D p);
	
	virtual OSErr		TextRead(char *path,char *topFilePath);
	virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg);
	//virtual OSErr		ScanFileForTimes(char *path,PtCurTimeDataHdl *timeDataH,Boolean setStartTime);
	virtual long		GetNumTimesInFile();
	OSErr				ReadHeaderLines(char *path, WorldRect *bounds);
	OSErr			ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH, char *pathOfInputfile);
	virtual void		DisposeTimeHdl();
	virtual OSErr 		CheckAndScanFile(char *errmsg, const Seconds& model_time);	
	virtual OSErr		GetStartTime(Seconds *startTime);	// switch this to GetTimeValue
	virtual OSErr		GetEndTime(Seconds *endTime);
};

class TimeGridCurTri_c : virtual public TimeGridCurRect_c
{
public:
	
	FLOATH fDepthsH;
	DepthDataInfoH fDepthDataInfo;
	PTCurVariables fVar2;
	
	TimeGridCurTri_c();
	//virtual	~TimeGridCurTri_c() { Dispose (); }
	
	//virtual void	Dispose() { return; }
	
	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDCURTRI; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEGRIDCURTRI) return TRUE; return TimeGridCurRect_c::IAm(id); }
	
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint3D p);
	VelocityRec 		GetScaledPatValue3D(const Seconds& model_time, InterpolationVal interpolationVal,float depth);
	
	virtual OSErr		TextRead(char *path,char *topFilePath);
	virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg);
	OSErr				ReadHeaderLine(char *s);
	OSErr				ReadPtCurVertices(CHARH fileBufH,long *line,LongPointHdl *pointsH,FLOATH *bathymetryH,char* errmsg,long numPoints);
	
	long					GetNumDepths(void);
	void 					GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2);
};

class TimeGridWindRect_c : virtual public TimeGridVel_c
{
public:
	
	TimeGridWindRect_c();
	//virtual	~TimeGridWindRect_c() { Dispose (); }
	
	//virtual void	Dispose() { return; }
	
	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDWINDRECT; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEGRIDWINDRECT) return TRUE; return TimeGridVel_c::IAm(id); }
	
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint3D p);
	
	virtual OSErr		TextRead(char *path,char *topFilePath);
	virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg);
	
};

class TimeGridWindCurv_c : virtual public TimeGridWindRect_c
{
public:
	
	LONGH fVerdatToNetCDFH;	// for curvilinear
	WORLDPOINTFH fVertexPtsH;		// for curvilinear, all vertex points from file
	
	
	TimeGridWindCurv_c();
	//virtual	~TimeGridWindCurv_c() { Dispose (); }
	
	//virtual void	Dispose() { return; }
	
	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDWINDCURV; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEGRIDWINDCURV) return TRUE;  return TimeGridWindRect_c::IAm(id); }
	
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint3D p);
	
	virtual long 		GetVelocityIndex(WorldPoint wp);
	virtual LongPoint 	GetVelocityIndices(WorldPoint wp);

	virtual OSErr		TextRead(char *path,char *topFilePath);
	virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg);
	
	OSErr 				ReorderPoints(char* errmsg); 
	OSErr				GetLatLonFromIndex(long iIndex, long jIndex, WorldPoint *wp);

	virtual	OSErr 	ReadTopology(char* path);
};
/*class TimeGridWindRectASCII_c : virtual public TimeGridVel_c
{
public:
	// code goes here, build off of TimeGridCurRect_c ??
	PtCurTimeDataHdl fTimeDataHdl;	
	short fUserUnits;
	
	TimeGridWindRectASCII_c();
	//virtual	~TimeGridCurRect_c() { Dispose (); }
	
	//virtual void	Dispose() { return; }
	
	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDWINDRECTASCII; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEGRIDWINDRECTASCII) return TRUE; return TimeGridVel_c::IAm(id); }
	
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint3D p);
	
	virtual OSErr		TextRead(char *path,char *topFilePath);
	virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg);
	//virtual OSErr		ScanFileForTimes(char *path,PtCurTimeDataHdl *timeDataH,Boolean setStartTime);
	virtual long		GetNumTimesInFile();
	OSErr				ReadHeaderLines(char *path, WorldRect *bounds);
	OSErr			ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH, char *pathOfInputfile);
	virtual void		DisposeTimeHdl();
	virtual OSErr 		CheckAndScanFile(char *errmsg, const Seconds& model_time);	
	virtual OSErr		GetStartTime(Seconds *startTime);	// switch this to GetTimeValue
	virtual OSErr		GetEndTime(Seconds *endTime);
};*/

//#endif	
#endif
