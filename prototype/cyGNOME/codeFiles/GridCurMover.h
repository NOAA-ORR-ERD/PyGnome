
#ifndef __GRIDCURMOVER__
#define __GRIDCURMOVER__

#include "GridVel.h"
#include "PtCurMover.h"

Boolean IsGridCurTimeFile (char *path, short *selectedUnits);


class GridCurMover : public TCATSMover
{
	public:
		long fNumRows;
		long fNumCols;
		PtCurTimeDataHdl fTimeDataHdl;
		LoadedData fStartData; 
		LoadedData fEndData;
		short fUserUnits;
		char fPathName[kMaxNameLen];
		char fFileName[kMaxNameLen];
		Boolean fOverLap;
		Seconds fOverLapStartTime;
		PtCurFileInfoH	fInputFilesHdl;

	public:
							GridCurMover (TMap *owner, char *name);
						   ~GridCurMover () { Dispose (); }

		//virtual OSErr		InitMover (); //  use TCATSMover version which sets grid ?
		virtual ClassID 	GetClassID () { return TYPE_GRIDCURMOVER; }
		virtual Boolean	IAm(ClassID id) { if(id==TYPE_GRIDCURMOVER) return TRUE; return TCATSMover::IAm(id); }
		virtual void		Dispose ();
		void 					DisposeLoadedData(LoadedData * dataPtr);	
		void 					ClearLoadedData(LoadedData * dataPtr);
		
		virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
		
		long 					GetVelocityIndex(WorldPoint p);
		VelocityRec			GetPatValue (WorldPoint p);
		VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99

		virtual OSErr		GetStartTime(Seconds *startTime);
		virtual OSErr		GetEndTime(Seconds *endTime);
		virtual double 	GetStartUVelocity(long index);
		virtual double 	GetStartVVelocity(long index);
		virtual double 	GetEndUVelocity(long index);
		virtual double 	GetEndVVelocity(long index);
		virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);

		long 					GetNumTimesInFile();
		long 					GetNumFiles();
		virtual OSErr 		CheckAndScanFile(char *errmsg);
		virtual Boolean 	CheckInterval(long &timeDataInterval);
		virtual OSErr	 	SetInterval(char *errmsg);
		virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
		virtual OSErr 		PrepareForModelStep();
		virtual void 		ModelStepIsDone();

		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		virtual OSErr		TextRead(char *path);
		OSErr 				ReadHeaderLines(char *path, WorldRect *bounds);
		OSErr 				ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
		OSErr 				ScanFileForTimes(char *path,PtCurTimeDataHdl *timeDataHdl,Boolean setStartTime);
		OSErr 				ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH, char *pathOfInputfile);

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

// build off TWindMover or off TMover with a lot of WindMover carried along ??
// for now new mover type but brought in under variable windmover
// what about constant wind case? parallel to gridcur?
class GridWindMover : public TWindMover
{
	public:
		Boolean 	bShowGrid;
		Boolean 	bShowArrows;
		char		fPathName[kMaxNameLen];
		char		fFileName[kPtCurUserNameLen]; // short file name
		
		long fNumRows;
		long fNumCols;
		TGridVel	*fGrid;	//VelocityH		grid; 
		PtCurTimeDataHdl fTimeDataHdl;
		LoadedData fStartData; 
		LoadedData fEndData;
		short	fUserUnits; 
		//float fFillValue;
		float fWindScale;	// not using this
		float fArrowScale;	// not using this
		Boolean fIsOptimizedForStep;

		Boolean fOverLap;
		Seconds fOverLapStartTime;
		PtCurFileInfoH	fInputFilesHdl;

	public:
							GridWindMover (TMap *owner, char* name);
						   ~GridWindMover () { Dispose (); }
		virtual ClassID 	GetClassID () { return TYPE_GRIDWINDMOVER; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_GRIDWINDMOVER) return TRUE; return TWindMover::IAm(id); }
		virtual void		Dispose ();
		void 					DisposeLoadedData(LoadedData * dataPtr);	
		void 					ClearLoadedData(LoadedData * dataPtr);
		
		virtual OSErr 		PrepareForModelStep();
		virtual void 		ModelStepIsDone();
		virtual WorldPoint3D 	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);

		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb);  // read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position
		
		// list display methods
		virtual long		GetListLength ();
		virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
//		virtual OSErr 		AddItem (ListItem item);
		virtual OSErr		SettingsItem (ListItem item);
		virtual OSErr 		DeleteItem (ListItem item);

	public:
		virtual long 		GetVelocityIndex(WorldPoint p);
		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);

		long 					GetNumTimesInFile();
		long 					GetNumFiles();
		//virtual OSErr		GetStartTime(Seconds *startTime);
		//virtual OSErr		GetEndTime(Seconds *endTime);

		virtual Boolean 	CheckInterval(long &timeDataInterval);
		virtual OSErr	 	SetInterval(char *errmsg);

		virtual OSErr		TextRead(char *path);
		OSErr 				ReadHeaderLines(char *path, WorldRect *bounds);
		virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
		OSErr 				ScanFileForTimes(char *path, PtCurTimeDataHdl *timeDataH,Boolean setStartTime);
		OSErr 				ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH, char *pathOfInputfile);

		virtual void		Draw (Rect r, WorldRect view);
		virtual Boolean	DrawingDependsOnTime(void);
		
};

//OSErr	GridWindSettingsDialog(GridWindMover *mover, TMap *owner,Boolean bAddMover,DialogPtr parentWindow);
Boolean IsGridWindFile(char *path,short *selectedUnits);

#endif //  __GRIDCURMOVER__
