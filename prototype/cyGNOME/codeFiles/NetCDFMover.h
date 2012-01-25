
#ifndef __NETCDFMOVER__
#define __NETCDFMOVER__

#include "GridVel.h"
#include "PtCurMover.h"

enum { REGULAR=1, REGULAR_SWAFS, CURVILINEAR, TRIANGULAR, REGRIDDED};	// maybe eliminate regridded option

Seconds RoundDateSeconds(Seconds timeInSeconds);
PtCurMap* GetPtCurMap(void);

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

class NetCDFMover : public TCurrentMover
{
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

	public:
							NetCDFMover (TMap *owner, char *name);
						   ~NetCDFMover () { Dispose (); }

		virtual OSErr		InitMover (); //  use TCATSMover version which sets grid ?
		virtual ClassID 	GetClassID () { return TYPE_NETCDFMOVER; }
		virtual Boolean	IAm(ClassID id) { if(id==TYPE_NETCDFMOVER) return TRUE; return TCurrentMover::IAm(id); }
		virtual void		Dispose ();
		void 					DisposeLoadedData(LoadedData * dataPtr);	
		void 					ClearLoadedData(LoadedData * dataPtr);
		void 					DisposeAllLoadedData();
		virtual	OSErr 	ReplaceMover();
		
		virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
		
		virtual long 		GetVelocityIndex(WorldPoint p);
		virtual LongPoint 	GetVelocityIndices(WorldPoint wp);
		VelocityRec			GetPatValue (WorldPoint p);
		VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99

		long 					GetNumTimesInFile();
		long 					GetNumFiles();
		long 					GetNumDepths(void);
		virtual long 		GetNumDepthLevels();
		virtual long 		GetNumDepthLevelsInFile();	// eventually get rid of this
		//virtual DepthValuesSetH 	GetDepthProfileAtPoint(WorldPoint refPoint) {return nil;}
		virtual OSErr 	GetDepthProfileAtPoint(WorldPoint refPoint, long timeIndex, DepthValuesSetH *profilesH) {*profilesH=nil; return 0;}
		void 					GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2);
		float 				GetMaxDepth();
		virtual float		GetArrowDepth() {return fVar.arrowDepth;}

		virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	

		Seconds 				GetTimeValue(long index);
		virtual OSErr		GetStartTime(Seconds *startTime);
		virtual OSErr		GetEndTime(Seconds *endTime);
		virtual double 	GetStartUVelocity(long index);
		virtual double 	GetStartVVelocity(long index);
		virtual double 	GetEndUVelocity(long index);
		virtual double 	GetEndVVelocity(long index);
		virtual double	GetDepthAtIndex(long depthIndex, double totalDepth);
		virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
		float		GetTotalDepth(WorldPoint refPoint, long triNum);

		virtual OSErr 		CheckAndScanFile(char *errmsg);
		virtual Boolean 	CheckInterval(long &timeDataInterval);
		virtual OSErr	 	SetInterval(char *errmsg);
		virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
		virtual OSErr 		PrepareForModelStep();
		virtual void 		ModelStepIsDone();

		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		//virtual OSErr		TextRead(char *path,TMap **newMap);
		virtual OSErr		TextRead(char *path,TMap **newMap,char *topFilePath);
		virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
		OSErr 				ScanFileForTimes(char *path,Seconds ***timeH,Boolean setStartTime);
		//OSErr 				ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH);
		OSErr 				ReadInputFileNames(char *fileNamesPath);

		// list display methods
		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
		
		virtual void		DrawContourScale(Rect r, WorldRect view);
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

		OSErr					SaveAsNetCDF(char *path,double *lats, double *lons);	// for testing -  may use in CATS eventually
		//OSErr					SaveAsVis5d(char *path,double *lats, double *lons);	// for testing 
		//OSErr					SaveAsVis5d(double endLat, double startLon, double dLat, double dLon);	// for testing 

		//OSErr 				SetTimesForVis5d(int *timestamp, int *datestamp);
};

typedef struct GridCellInfo
{
	long cellNum;	//index of rectangle
	long 	topLeft;	//top left index
	long 	topRight;	//top right index
	long 	bottomLeft;	//bottom left index
	long 	bottomRight;	//bottom right index
} GridCellInfo, *GridCellInfoPtr, **GridCellInfoHdl;

typedef struct SegInfo
{
	long pt1;	//index of first point
	long 	pt2;	//index of second point
	long 	islandNumber;	//land block
	Boolean 	isWater;	//land/water boudary
} SegInfo, *SegInfoPtr, **SegInfoHdl;

class NetCDFMoverCurv : public NetCDFMover
{
	public:
		LONGH fVerdatToNetCDFH;	// for curvilinear
		WORLDPOINTFH fVertexPtsH;		// for curvilinear, all vertex points from file
		LONGH fVerdatToNetCDFH_2;	// for curvilinear

	public:
							NetCDFMoverCurv (TMap *owner, char *name);
						   ~NetCDFMoverCurv () { Dispose (); }

		virtual ClassID 	GetClassID () { return TYPE_NETCDFMOVERCURV; }
		virtual Boolean	IAm(ClassID id) { if(id==TYPE_NETCDFMOVERCURV) return TRUE; return NetCDFMover::IAm(id); }
		virtual Boolean		IAmA3DMover();
		virtual void		Dispose ();
		
		LongPointHdl		GetPointsHdl();
		virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
		virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);

		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		//virtual OSErr		TextRead(char *path,TMap **newMap);
		virtual OSErr		TextRead(char *path,TMap **newMap,char *topFilePath);
		virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
 		
		virtual	OSErr 	ReadTopology(char* path, TMap **newMap);
		virtual	OSErr 	ExportTopology(char* path);
		//OSErr 				ReadTransposeArray(CHARH fileBufH,long *line,LONGH *transposeArray,long numPts,char* errmsg);

		long 				CheckSurroundingPoints(LONGH maskH, long row, long col) ;
		Boolean 			InteriorLandPoint(LONGH maskH, long row, long col); 
		Boolean 			ThereIsAdjacentLand2(LONGH maskH, VelocityFH velocityH, long row, long col) ;
		Boolean 			ThereIsALowerLandNeighbor(LONGH maskH, long *lowerPolyNum, long row, long col) ;
		void 				ResetMaskValues(LONGH maskH,long landBlockToMerge,long landBlockToJoin);
		OSErr 				NumberIslands(LONGH *islandNumberH, VelocityFH velocityH,LONGH landWaterInfo,long *numIslands);
		OSErr 				ReorderPoints(VelocityFH velocityH, TMap **newMap, char* errmsg); 
		OSErr 				ReorderPointsNoMask(VelocityFH velocityH, TMap **newMap, char* errmsg); 
		OSErr 				ReorderPointsNoMask2(VelocityFH velocityH, TMap **newMap, char* errmsg); 

		// list display methods
		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
		
		//double GetTopDepth(long depthIndex, double totalDepth);
		//double GetBottomDepth(long depthIndex, double totalDepth);

		virtual long 		GetVelocityIndex(WorldPoint wp);
		virtual LongPoint 	GetVelocityIndices(WorldPoint wp);
		OSErr 				GetLatLonFromIndex(long iIndex, long jIndex, WorldPoint *wp);
		virtual long 		GetNumDepthLevels();
		virtual OSErr 		GetDepthProfileAtPoint(WorldPoint refPoint, long timeIndex, DepthValuesSetH *profilesH);
		void 				GetDepthIndices(long ptIndex, float depthAtPoint, float totalDepth, long *depthIndex1, long *depthIndex2);
		float		GetTotalDepthFromTriIndex(long triIndex);
		float		GetTotalDepth(WorldPoint refPoint,long ptIndex);
		virtual void		Draw (Rect r, WorldRect view);
		virtual void		DrawContourScale(Rect r, WorldRect view);
		
		//virtual long		GetListLength ();
		//virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		//virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		//virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
		//virtual OSErr 		AddItem (ListItem item);
		//virtual OSErr 		SettingsItem (ListItem item);
		//virtual OSErr 		DeleteItem (ListItem item);
		
		//virtual OSErr 		SettingsDialog();

};

// may eventually derive from NetCDFMoverCurv once things stabilize
//class NetCDFMoverTri : public NetCDFMover
class NetCDFMoverTri : public NetCDFMoverCurv
{
	public:
		//LONGH fVerdatToNetCDFH;	
		//WORLDPOINTFH fVertexPtsH;	// may not need this if set pts in dagtree	
		long fNumNodes;
		long fNumEles;	//for now, number of triangles
		Boolean bVelocitiesOnTriangles;

	public:
							NetCDFMoverTri (TMap *owner, char *name);
						   ~NetCDFMoverTri () { Dispose (); }

		virtual ClassID 	GetClassID () { return TYPE_NETCDFMOVERTRI; }
		//virtual Boolean	IAm(ClassID id) { if(id==TYPE_NETCDFMOVERTRI) return TRUE; return NetCDFMover::IAm(id); }
		virtual Boolean	IAm(ClassID id) { if(id==TYPE_NETCDFMOVERTRI) return TRUE; return NetCDFMoverCurv::IAm(id); }
		virtual void		Dispose ();
		
		LongPointHdl		GetPointsHdl();
		Boolean 				VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
		virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
		VelocityRec 	GetMove3D(InterpolationVal interpolationVal,float depth);
		void 	GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2);

		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to current position

		//virtual OSErr		TextRead(char *path,TMap **newMap);
		virtual OSErr		TextRead(char *path,TMap **newMap,char *topFilePath);
		virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 

		virtual	OSErr 	ReadTopology(char* path, TMap **newMap);
		virtual 	OSErr 	ExportTopology(char* path);

		//OSErr 				ReorderPoints(TMap **newMap, short *bndry_indices, short *bndry_nums, short *bndry_type, long numBoundaryPts); 
		OSErr 				ReorderPoints(TMap **newMap, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts); 
		OSErr				ReorderPoints2(TMap **newMap, long *bndry_indices, long *bndry_nums, long *bndry_type, long numBoundaryPts, long *tri_verts, long *tri_neighbors, long ntri);

		virtual long 		GetNumDepthLevels();
		virtual OSErr 		GetDepthProfileAtPoint(WorldPoint refPoint, long timeIndex, DepthValuesSetH *profilesH) {*profilesH=nil; return 0;}
		float		GetTotalDepth(WorldPoint refPoint, long triNum);

		// list display methods
		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
		
		virtual void		Draw (Rect r, WorldRect view);
		//virtual void		DrawContourScale(Rect r, WorldRect view);
		
		//virtual long		GetListLength ();
		//virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		//virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		//virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
		//virtual OSErr 		AddItem (ListItem item);
		//virtual OSErr 		SettingsItem (ListItem item);
		//virtual OSErr 		DeleteItem (ListItem item);
		
		//virtual OSErr 		SettingsDialog();

};
// build off TWindMover or off TMover with a lot of WindMover carried along ??
// for now new mover type but brought in under variable windmover
class NetCDFWindMover : public TWindMover
{
	public:
		Boolean 	bShowGrid;
		Boolean 	bShowArrows;

		////// start: new fields to support multi-file NetCDFPathsFile
		Boolean fOverLap;
		Seconds fOverLapStartTime;
		PtCurFileInfoH	fInputFilesHdl; 
		////// end:  multi-file fields

		char		fPathName[kMaxNameLen];
		char		fFileName[kPtCurUserNameLen]; // short file name
		//char		fFileName[kMaxNameLen]; // short file name - might want to allow longer names
		
		long fNumRows;
		long fNumCols;
		//NetCDFVariables fVar;
		TGridVel	*fGrid;	//VelocityH		grid; 
		//PtCurTimeDataHdl fTimeDataHdl;
		Seconds **fTimeHdl;
		LoadedData fStartData; 
		LoadedData fEndData;
		short fUserUnits;
		//double fFillValue;
		float fFillValue;
		float fWindScale;
		float fArrowScale;
		long fTimeShift;		// to convert GMT to local time
		Boolean fAllowExtrapolationOfWinds;
		Boolean fIsOptimizedForStep;

	public:
							NetCDFWindMover (TMap *owner, char* name);
						   ~NetCDFWindMover () { Dispose (); }
		virtual ClassID 	GetClassID () { return TYPE_NETCDFWINDMOVER; }
		virtual Boolean		IAm(ClassID id) { if(id==TYPE_NETCDFWINDMOVER) return TRUE; return TWindMover::IAm(id); }
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
		virtual LongPoint 		GetVelocityIndices(WorldPoint wp); /*{LongPoint lp = {-1,-1}; printError("GetVelocityIndices not defined for windmover"); return lp;}*/
		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);

		long 					GetNumTimesInFile();
		Seconds 				GetTimeValue(long index);
		//virtual OSErr		GetStartTime(Seconds *startTime);
		//virtual OSErr		GetEndTime(Seconds *endTime);
		virtual OSErr		GetStartTime(Seconds *startTime);
		virtual OSErr		GetEndTime(Seconds *endTime);
		virtual double 	GetStartUVelocity(long index);
		virtual double 	GetStartVVelocity(long index);
		virtual double 	GetEndUVelocity(long index);
		virtual double 	GetEndVVelocity(long index);
		virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);

		virtual Boolean 	CheckInterval(long &timeDataInterval);
		virtual OSErr	 	SetInterval(char *errmsg);

		virtual WorldRect GetGridBounds(){return fGrid->GetBounds();}	

		virtual OSErr		TextRead(char *path);
		virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 

		virtual void		Draw (Rect r, WorldRect view);
		virtual Boolean	DrawingDependsOnTime(void);


		//////////////////		
		OSErr ReadInputFileNames(char *fileNamesPath);
		void DisposeAllLoadedData();
		long GetNumFiles();
		OSErr CheckAndScanFile(char *errmsg);
		OSErr ScanFileForTimes(char *path,Seconds ***timeH,Boolean setStartTime);
		
};

OSErr			NetCDFWindSettingsDialog(NetCDFWindMover *mover, TMap *owner,Boolean bAddMover,WindowPtr parentWindow);

class NetCDFWindMoverCurv : public NetCDFWindMover
{
	public:
		LONGH fVerdatToNetCDFH;	// for curvilinear
		WORLDPOINTFH fVertexPtsH;		// for curvilinear, all vertex points from file

	public:
							NetCDFWindMoverCurv (TMap *owner, char *name);
						   ~NetCDFWindMoverCurv () { Dispose (); }

		virtual ClassID 	GetClassID () { return TYPE_NETCDFWINDMOVERCURV; }
		virtual Boolean	IAm(ClassID id) { if(id==TYPE_NETCDFWINDMOVERCURV) return TRUE; return NetCDFWindMover::IAm(id); }
		virtual void		Dispose ();
		
		LongPointHdl		GetPointsHdl();
		virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
		virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);

		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		virtual OSErr		TextRead(char *path,TMap **newMap);
		virtual OSErr 		ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
 		
		virtual OSErr 	 	ReadTopology(char* path, TMap **newMap);
		virtual OSErr 		ExportTopology(char* path);

		long 					CheckSurroundingPoints(LONGH maskH, long row, long col) ;
		Boolean 				InteriorLandPoint(LONGH maskH, long row, long col); 
		Boolean 				ThereIsAdjacentLand2(LONGH maskH, VelocityFH velocityH, long row, long col) ;
		Boolean 				ThereIsALowerLandNeighbor(LONGH maskH, long *lowerPolyNum, long row, long col) ;
		void 					ResetMaskValues(LONGH maskH,long landBlockToMerge,long landBlockToJoin);
		OSErr 				NumberIslands(LONGH *islandNumberH, VelocityFH velocityH,LONGH landWaterInfo,long *numIslands);
		OSErr 				ReorderPoints(VelocityFH velocityH, TMap **newMap, char* errmsg); 

		// list display methods
		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
		
		Seconds 				GetTimeValue(long index);
		virtual long 		GetVelocityIndex(WorldPoint wp);
		virtual LongPoint 		GetVelocityIndices(WorldPoint wp);  /*{LongPoint lp = {-1,-1}; printError("GetVelocityIndices not defined for windmover"); return lp;}*/
		OSErr 				GetLatLonFromIndex(long iIndex, long jIndex, WorldPoint *wp);
		virtual void		Draw (Rect r, WorldRect view);
		
		//virtual long		GetListLength ();
		//virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		//virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		//virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
		//virtual OSErr 		AddItem (ListItem item);
		//virtual OSErr 		SettingsItem (ListItem item);
		//virtual OSErr 		DeleteItem (ListItem item);
		
		//virtual OSErr 		SettingsDialog();

};
#endif //  __NETCDFMOVER__
