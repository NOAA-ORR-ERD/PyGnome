
#ifndef __TIDECURCYCLEMOVER__
#define __TIDECURCYCLEMOVER__

#include "GridVel.h"
#include "PtCurMover.h"

Boolean IsTideCurCycleFile (char *path, short *gridType);


class TideCurCycleMover : public TCATSMover
{
	public:
		//long fNumRows;
		//long fNumCols;
		//PtCurTimeDataHdl fTimeDataHdl;
		Seconds **fTimeHdl;
		LoadedData fStartData; 
		LoadedData fEndData;
		float fFillValue;
		float fDryValue;
		short fUserUnits;
		char fPathName[kMaxNameLen];
		char fFileName[kMaxNameLen];
		LONGH fVerdatToNetCDFH;		// these two fields will be in curvilinear if we extend there
		WORLDPOINTFH fVertexPtsH;	// may not need this if set pts in dagtree	
		long fNumNodes;
		short fPatternStartPoint;	// maxflood, maxebb, etc
		float fTimeAlpha;
		char fTopFilePath[kMaxNameLen];

	public:
							TideCurCycleMover (TMap *owner, char *name);
						   ~TideCurCycleMover () { Dispose (); }

		//virtual OSErr		InitMover (); //  use TCATSMover version which sets grid ?
		virtual ClassID 	GetClassID () { return TYPE_TIDECURCYCLEMOVER; }
		virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIDECURCYCLEMOVER) return TRUE; return TCATSMover::IAm(id); }
		virtual void		Dispose ();
		void 					DisposeLoadedData(LoadedData * dataPtr);	
		void 					ClearLoadedData(LoadedData * dataPtr);
		
		virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
		
		LongPointHdl 		GetPointsHdl();
		//long 					GetVelocityIndex(WorldPoint p);
		VelocityRec			GetPatValue (WorldPoint p);
		VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99

		/*virtual OSErr		GetStartTime(Seconds *startTime);
		virtual OSErr		GetEndTime(Seconds *endTime);*/
		virtual double 	GetStartUVelocity(long index);
		virtual double 	GetStartVVelocity(long index);
		virtual double 	GetEndUVelocity(long index);
		virtual double 	GetEndVVelocity(long index);
		virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
		virtual OSErr 		ComputeVelocityScale();

		Boolean 				IsDryTriangle(long index1, long index2, long index3, float timeAlpha);
		Boolean 				IsDryTri(long triIndex);
		VelocityRec 		GetStartVelocity(long index, Boolean *isDryPt);
		VelocityRec 		GetEndVelocity(long index, Boolean *isDryPt);

		long 					GetNumTimesInFile();
		virtual Boolean 	CheckInterval(long &timeDataInterval);
		virtual OSErr	 	SetInterval(char *errmsg);
		virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
		virtual OSErr 		PrepareForModelStep();
		virtual void 		ModelStepIsDone();

		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		//virtual OSErr		TextRead(char *path, TMap **newMap);
		virtual OSErr		TextRead(char *path, TMap **newMap, char *topFilePath);
		OSErr 				ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 

		virtual	OSErr 	ReadTopology(char* path, TMap **newMap);
		virtual 	OSErr 	ExportTopology(char* path);

		OSErr 				ReorderPoints(TMap **newMap, short *bndry_indices, short *bndry_nums, short *bndry_type, long numBoundaryPts); 

		// list display methods
		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
		
		virtual void		Draw (Rect r, WorldRect view);
		virtual Boolean	DrawingDependsOnTime(void);
		
		//virtual long		GetListLength ();
		virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		//virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
		//virtual OSErr 		AddItem (ListItem item);
		virtual OSErr 		SettingsItem (ListItem item);
		virtual OSErr 		DeleteItem (ListItem item);
		
		
		//virtual OSErr 		SettingsDialog();

};


#endif //  __TIDECURCYCLEMOVER__
