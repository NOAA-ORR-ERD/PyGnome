#ifndef __RANDOMGRIDMOVER__
#define __RANDOMGRIDMOVER__

#include "GridVel.h"
#include "PtCurMover.h"

// classes for CATS style horizontally varying diffusion mover and gridcur style
// not sure if the diffusion would be fudged in CATS on a triangle grid or on a regular grid
// if regular would need to ensure it was bigger than the current grid or decide what to do
// in areas without coverage
// also the optimize stuff may change
///////////////////////////////////////////////////////////////////////////

typedef struct {
	Boolean isOptimizedForStep;
	Boolean isFirstStep;
	double 	value;
	double 	uncertaintyValue;
} TR_OPTIMZE;

class TRandom : public TMover
{
	public:
		double fDiffusionCoefficient; //cm**2/s
		TR_OPTIMZE fOptimize; // this does not need to be saved to the save file
		double fUncertaintyFactor;		// multiplicative factor applied when uncertainty is on
		Boolean bUseDepthDependent;
	
	public:
							TRandom (TMap *owner, char *name);
		virtual ClassID 	GetClassID () { return TYPE_RANDOMMOVER; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_RANDOMMOVER) return TRUE; return TMover::IAm(id); }
		
		virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, const Seconds&, const Seconds&, bool); // AH 07/10/2012
	
		virtual void 		ModelStepIsDone();
		virtual WorldPoint3D 	GetMove (const Seconds& start_time, const Seconds& stop_time, const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
		
		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb);  // read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		// list display methods
		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
		virtual void		Draw(Rect r, WorldRect view) { }
		virtual long		GetListLength();
		virtual ListItem 	GetNthListItem(long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick(ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled(ListItem item, short buttonID);
//		virtual OSErr 		AddItem(ListItem item);
		virtual OSErr 		SettingsItem(ListItem item);
		virtual OSErr 		DeleteItem(ListItem item);
};

///////////////////////////////////////////////////////////////////////////
class TGridVel;
///////////////////////////////////////////////////////////////////////////

#define		kUCode			0					// supplied to UorV routine
#define		kVCode			1

typedef struct {
	Boolean isOptimizedForStep;
	Boolean isFirstStep;
	double 	value;
} TCM_OPTIMZE;
// if build off Random rather than Current as CATS does will lose uncertainty
// stuff. Not sure what to do uncertaintywise anyway...
class TRandomCATSMover : public TRandomMover
{
	public:
		TGridVel		*fGrid;					//VelocityH		grid; 
		double 			scaleValue; 			// constant value to match at refP
		Boolean			bUncertaintyPointOpen;
		Boolean 		bTimeFileOpen;
		Boolean			bTimeFileActive;		// active / inactive flag
		Boolean 		bShowGrid;
		Boolean 		bShowArrows;
		double 			arrowScale;
		double			fEddyDiffusion;			// cm**2/s minimum eddy velocity for uncertainty
		double			fEddyV0;			//  in m/s, used for cutoff of minimum eddy for uncertainty
	public:
		TCM_OPTIMZE fOptimize; // this does not need to be saved to the save file

	public:
							TRandomCATSMover (TMap *owner, char *name);
						   ~TRandomCATSMover () { Dispose (); }
		virtual OSErr		InitMover (TGridVel *grid, WorldPoint p);
		virtual ClassID 	GetClassID () { return TYPE_RANDOMCATSMOVER; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_RANDOMCATSMOVER) return TRUE; return TRandomMover::IAm(id); }
		virtual void		Dispose ();
		virtual Boolean 	OkToAddToUniversalMap();
		virtual	OSErr 		ReplaceMover();
		// other uncertainty functions?
		virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
		
		VelocityRec			GetPatValue (WorldPoint p);
		VelocityRec 		GetScaledPatValue(const Seconds& start_time, const Seconds& stop_time, const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
		virtual WorldPoint3D	GetMove (const Seconds& start_time, const Seconds& stop_time, const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);
		virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, const Seconds&, const Seconds&, bool); // AH 07/10/2012
	
		virtual void 		ModelStepIsDone();

		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		// list display methods
		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
		virtual void		Draw (Rect r, WorldRect view);
		virtual Boolean		VelocityStrAtPoint(WorldPoint3D wp, char *velStr);
		virtual long		GetListLength ();
		virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
//		virtual OSErr 		AddItem (ListItem item);
		virtual OSErr 		SettingsItem (ListItem item);
		virtual OSErr 		DeleteItem (ListItem item);
};

///////////////////////////////////////////////////////////////////////////
// if build off Random rather than Wind as GridWind does will lose uncertainty
// stuff. Not sure what to do uncertaintywise anyway...
// Or may want netcdf for this option
class TRandomGridMover : public TRandomMover
{
	public:
		Boolean 	bShowGrid;
		Boolean 	bShowArrows;
		char		fPathName[kMaxNameLen];
		char		fFileName[kPtCurUserNameLen]; // short file name
		
		// TWindMover fields, probably won't use, but need something for uncertainty
		//double fSpeedScale;
		//double fAngleScale;
		//double fMaxSpeed;
		//double fMaxAngle;
		// uncertainstarttime and duration are ok since they are part of TMover

		long fNumRows;
		long fNumCols;
		TGridVel	*fGrid;	//VelocityH		grid; 
		PtCurTimeDataHdl fTimeDataHdl;
		LoadedData fStartData; 
		LoadedData fEndData;
		short	fUserUnits; 
		//float fFillValue;
		float fDiffusionScale;	
		float fArrowScale;	
		Boolean fIsOptimizedForStep;

		//Boolean fOverLap;
		//Seconds fOverLapStartTime;
		//PtCurFileInfoH	fInputFilesHdl;

	public:
							TRandomGridMover (TMap *owner, char* name);
						   ~TRandomGridMover () { Dispose (); }
		virtual ClassID 	GetClassID () { return TYPE_RANDOMGRIDMOVER; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_RANDOMGRIDMOVER) return TRUE; return TRandomMover::IAm(id); }
		virtual void		Dispose ();
		void 					DisposeLoadedData(LoadedData * dataPtr);	
		void 					ClearLoadedData(LoadedData * dataPtr);
		
		virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, const Seconds&, const Seconds&, bool); // AH 07/10/2012
	
		virtual void 		ModelStepIsDone();
		virtual WorldPoint3D 	GetMove (const Seconds& start_time, const Seconds& stop_time, const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *theLE,LETYPE leType);

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
		OSErr 				ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH, char * pathOfInputfile);

		virtual void		Draw (Rect r, WorldRect view);
		virtual Boolean	DrawingDependsOnTime(void);
		
};
//DiffusionSettingsDialog
//OSErr	GridDiffusionSettingsDialog(GridWindMover *mover, TMap *owner,Boolean bAddMover,DialogPtr parentWindow);
Boolean IsRandomGridFile(char *path,short *selectedUnits);

#endif //  __RANDOMGRIDMOVER__
