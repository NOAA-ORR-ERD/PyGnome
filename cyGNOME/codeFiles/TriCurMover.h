
#ifndef __TRICURMOVER__
#define __TRICURMOVER__

#include "DagTree.h"
#include "my_build_list.h"
#include "GridVel.h"

Boolean IsTriCurFile (char *path);
Boolean IsTriCurVerticesHeaderLine(char *s, long* numPts);

enum {ONELAYER_CONSTDENS=1, ONELAYER_VARDENS, TWOLAYER_CONSTDENS, TWOLAYER_VARDENS};	// gridtypes

typedef struct {
	char		curFilePathName[kMaxNameLen]; // currents
	char		sshFilePathName[kMaxNameLen]; // sea surface height
	char		pycFilePathName[kMaxNameLen]; // pycnocline depth
	char		lldFilePathName[kMaxNameLen];	// lower level density
	char		uldFilePathName[kMaxNameLen]; // upper level density
	short 	modelType;	// 1 layer constant density, 1 layer variable density, 2 layer constant density, 2 layer variable density	
	double 	scaleVel;	// cm/s
	double 	bottomBLThickness;	// cm
	double 	upperEddyViscosity;	// cm^2/s
	double 	lowerEddyViscosity;	// cm^2/s
	double 	upperLevelDensity;	// gm/cm^3
	double 	lowerLevelDensity;	// gm/cm^3
} BaromodesParameters;

class TriCurMover : public TCurrentMover
{
	public:
		PTCurVariables fVar;	// not sure if this is really necessary
		BaromodesParameters fInputValues;
		TGridVel	*fGrid;	
		PtCurTimeDataHdl fTimeDataHdl;
		LoadedData fStartData; 
		LoadedData fEndData;
		FLOATH fDepthsH;	
		DepthDataInfoH fDepthDataInfo;	// triangle info?
		Boolean fIsOptimizedForStep;
		//Boolean fOverLap;
		//Seconds fOverLapStartTime;
		//PtCurFileInfoH	fInputFilesHdl;
		Rect fLegendRect;
		Boolean bShowDepthContours;
		Boolean bShowDepthContourLabels;

	public:
							TriCurMover (TMap *owner, char *name);
						   ~TriCurMover () { Dispose (); }

		virtual OSErr		InitMover ();
		virtual ClassID 	GetClassID () { return TYPE_TRICURMOVER; }
		virtual Boolean	IAm(ClassID id) { if(id==TYPE_TRICURMOVER) return TRUE; return TCurrentMover::IAm(id); }
		virtual Boolean		IAmA3DMover(){return true;}
		virtual void		Dispose ();
		void 					DisposeLoadedData(LoadedData * dataPtr);	
		void 					ClearLoadedData(LoadedData * dataPtr);
		
		virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
		

		VelocityRec			GetPatValue (WorldPoint p);
		VelocityRec 		GetScaledPatValue(WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99

		virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);

		long 					GetNumTimesInFile();
		//long 					GetNumFiles();
		long					GetNumDepths(void);
		float 				GetMaxDepth(void);
		virtual float		GetArrowDepth() {return fVar.arrowDepth;}
		virtual LongPointHdl GetPointsHdl();
		TopologyHdl 		GetTopologyHdl();
		long			 		WhatTriAmIIn(WorldPoint p);
		OSErr 				GetTriangleCentroid(long trinum, LongPoint *p);
		void 					GetDepthIndices(long ptIndex, float depthAtPoint, long *depthIndex1, long *depthIndex2);
		virtual Boolean 	CheckInterval(long &timeDataInterval);
		virtual OSErr	 	SetInterval(char *errmsg);
		virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
		virtual OSErr 		PrepareForModelStep();
		virtual void 		ModelStepIsDone();

		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		virtual OSErr		TextRead(char *path, TMap **newMap);
		//OSErr 				ReadTriCurVertices(CHARH fileBufH,long *line,LongPointHdl *pointsH,char* errmsg,long numPoints);
		OSErr 				ReadTriCurVertices(CHARH fileBufH,long *line,LongPointHdl *pointsH,FLOATH *totalDepthH,char* errmsg,long numPoints);
		OSErr 				ReadTriCurDepths(CHARH fileBufH,long *line,LongPointHdl *pointsH,char* errmsg,long numPoints);
		OSErr 				ReadHeaderLine(char *s);
		OSErr 				ReadBaromodesInputValues(CHARH fileBufH,long *line,BaromodesParameters *inputValues,char* errmsg,short modelType);
		OSErr 				ReadTimeData(long index,VelocityFH *velocityH, char* errmsg); 
		OSErr 				ScanFileForTimes(char *path,PtCurTimeDataHdl *timeDataHdl,Boolean setStartTime);
		OSErr 				ReadCentroidDepths(CHARH fileBufH,long *line,long numTris,/*FLOATH *centroidDepthsH,*/char* errmsg);
		OSErr 				ReadSigmaLevels(CHARH fileBufH,long *line,FLOATH *sigmaLevelsH,long numLevels,char* errmsg);
		//virtual OSErr 		CheckAndScanFile(char *errmsg);
		//OSErr 				ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH);

		//OSErr 				ReadTopology(char* path, TMap **newMap);
		//OSErr 				ExportTopology(char* path);

		// list display methods
		virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
		
		virtual void		Draw (Rect r, WorldRect view);
		virtual Boolean	DrawingDependsOnTime(void);
		void 					DrawContourScale(Rect r, WorldRect view);
		
		long					CreateDepthSlice(long triNum, float **depthSlice);
		OSErr 				CalculateVerticalGrid(LongPointHdl ptsH, FLOATH totalDepthH, TopologyHdl topH, long numTri,FLOATH sigmaLevels, long numSigmaLevels);

 		virtual long		GetListLength ();
		virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
		virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
		virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
		//virtual OSErr 		AddItem (ListItem item);
		virtual OSErr 		SettingsItem (ListItem item);
		virtual OSErr 		DeleteItem (ListItem item);
		
		
		virtual OSErr 		SettingsDialog();
		OSErr 				InputValuesDialog();

		void 				SetInputValues(BaromodesParameters inputValues) ;
};

#endif //  __TRICURMOVER__
