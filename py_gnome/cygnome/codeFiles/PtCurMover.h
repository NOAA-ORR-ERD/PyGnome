
#ifndef __PTCURMOVER__
#define __PTCURMOVER__

#include "DagTree/DagTree.h"
#include "my_build_list.h"
#include "GridVel.h"
#include "CurrentMover/TCurrentMover.h"

#ifdef pyGNOME
#define TMap Map_c
#endif

#include "Map/TMap.h"

Boolean IsPtCurFile (char *path);
Boolean IsPtCurVerticesHeaderLine(char *s, long* numPts, long* numLandPts);
OSErr ScanDepth (char *startChar, double *DepthPtr);
OSErr ScanVelocity (char *startChar, VelocityRec *VelocityPtr, long *scanLength);

#define kPtCurUserNameLen 64
#define UNASSIGNEDINDEX -1
#define BOTTOMINDEX -2	// below last data value, but above bottom
#define CONSTANTCURRENT 0
#define CONSTANTWIND 0

enum {TWO_D=1, BAROTROPIC, SIGMA, MULTILAYER, SIGMA_ROMS};	// gridtypes

enum {
		I_PTCURNAME = 0 ,
		I_PTCURACTIVE, 
		I_PTCURGRID, 
		I_PTCURARROWS,
	   I_PTCURSCALE,
		I_PTCURUNCERTAINTY,
		I_PTCURSTARTTIME,
		I_PTCURDURATION, 
		I_PTCURALONGCUR,
		I_PTCURCROSSCUR,
		I_PTCURMINCURRENT
		};

typedef struct {
	char		pathName[kMaxNameLen];
	char		userName[kPtCurUserNameLen]; // user name for the file
	double 	alongCurUncertainty;	
	double 	crossCurUncertainty;	
	double 	uncertMinimumInMPS;	
	double 	curScale;	
	double 	startTimeInHrs;	
	double 	durationInHrs;	
	//
	long		numLandPts; // 0 if boundary velocities defined, else set boundary velocity to zero
	long		maxNumDepths;
	short		gridType;
	double	bLayerThickness;
	//
	Boolean 	bShowGrid;
	Boolean 	bShowArrows;
	Boolean	bUncertaintyPointOpen;
	double 	arrowScale;
	double 	arrowDepth;	// depth level where velocities will be shown
} PTCurVariables;

typedef struct {
	long fileOffsetToStartOfData;
	long lengthOfData; // implicit from the next one
	Seconds time;
} PtCurTimeData,*PtCurTimeDataP,**PtCurTimeDataHdl;


typedef struct {
	long timeIndex;
	VelocityFH dataHdl; // numVertices
}  LoadedData,*LoadedDataP;

typedef struct {
	long timeIndex;
	VelocityFH3D dataHdl; // numVertices
}  LoadedData3D,*LoadedDataP3D;


typedef struct {
	float totalDepth;
	long indexToDepthData; 
	long numDepths;
}  DepthDataInfo,*DepthDataInfoP,**DepthDataInfoH;

typedef struct {
	char pathName[kMaxNameLen];
	Seconds startTime; 
	Seconds endTime;
}  PtCurFileInfo,*PtCurFileInfoP,**PtCurFileInfoH;


class PtCurMover : public TCurrentMover
{
	public:
		PTCurVariables fVar;
		TGridVel	*fGrid;	
		PtCurTimeDataHdl fTimeDataHdl;
		LoadedData fStartData; 
		LoadedData fEndData;
		FLOATH fDepthsH;
		DepthDataInfoH fDepthDataInfo;
		Boolean fIsOptimizedForStep;
		Boolean fOverLap;
		Seconds fOverLapStartTime;
		PtCurFileInfoH	fInputFilesHdl;

	public:
							PtCurMover (TMap *owner, char *name);
						   ~PtCurMover () { Dispose (); }

		virtual OSErr		InitMover ();
		virtual ClassID 	GetClassID () { return TYPE_PTCURMOVER; }
		virtual Boolean	IAm(ClassID id) { if(id==TYPE_PTCURMOVER) return TRUE; return TCurrentMover::IAm(id); }
		virtual void		Dispose ();
		void 					DisposeLoadedData(LoadedData * dataPtr);	
		void 					ClearLoadedData(LoadedData * dataPtr);
		
		virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
		

		VelocityRec			GetPatValue (WorldPoint p);
		VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99

		virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);

		long 					GetNumTimesInFile();
		long 					GetNumFiles();
		long					GetNumDepths(void);
		LongPointHdl 		GetPointsHdl();
		TopologyHdl 		GetTopologyHdl();
		long			 		WhatTriAmIIn(WorldPoint p);
		void 					GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2);
		virtual float		GetArrowDepth() {return fVar.arrowDepth;}
		virtual Boolean 	CheckInterval(long &timeDataInterval);
		virtual OSErr	 	SetInterval(char *errmsg);
		virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
		VelocityRec			GetMove3D(InterpolationVal interpolationVal,float depth);
		virtual OSErr 		PrepareForModelStep();
		virtual void 		ModelStepIsDone();

		virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	

		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		virtual OSErr		TextRead(char *path, TMap **newMap);
		OSErr 				ReadPtCurVertices(CHARH fileBufH,long *line,LongPointHdl *pointsH, FLOATH *totalDepth,char* errmsg,long numPoints);
		OSErr 				ReadHeaderLine(char *s);
		OSErr 				ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
		OSErr 				ScanFileForTimes(char *path,PtCurTimeDataHdl *timeDataHdl,Boolean setStartTime);
		virtual OSErr 		CheckAndScanFile(char *errmsg);
		OSErr 				ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH, char *pathOfInputfile);

		virtual	OSErr 	ReadTopology(char* path, TMap **newMap);
		virtual	OSErr 	ExportTopology(char* path);

		// list display methods
		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
		
		virtual void		Draw (Rect r, WorldRect view);
		virtual Boolean	DrawingDependsOnTime(void);
		
		virtual long		GetListLength ();
		virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
		//virtual OSErr 		AddItem (ListItem item);
		virtual OSErr 		SettingsItem (ListItem item);
		virtual OSErr 		DeleteItem (ListItem item);
		
		
		virtual OSErr 		SettingsDialog();

};

void CheckYear(short *year);

#endif //  __PTCURMOVER__
