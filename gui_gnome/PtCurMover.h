
#ifndef __PTCURMOVER__
#define __PTCURMOVER__

#include "TCurrentMover.h"
#include "DagTree.h"
#include "my_build_list.h"
#include "GridVel.h"
#include "PtCurMover_c.h"

Boolean IsPtCurFile (char *path);
Boolean IsPtCurVerticesHeaderLine(char *s, long* numPts, long* numLandPts);
OSErr ScanDepth (char *startChar, double *DepthPtr);
void CheckYear(short *year);


class PtCurMover : virtual public PtCurMover_c,  public TCurrentMover
{
	public:
							PtCurMover (TMap *owner, char *name);
						   ~PtCurMover () { Dispose (); }

		virtual OSErr		InitMover ();

		virtual void		Dispose ();
	

		virtual WorldPoint3D	GetMove (Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
		VelocityRec			GetMove3D(InterpolationVal interpolationVal,float depth);
		virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);

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
		virtual Boolean 	CheckInterval(long &timeDataInterval);
		virtual OSErr	 	SetInterval(char *errmsg);
		virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, const Seconds&, bool); // AH 04/16/12
		virtual OSErr 		SettingsDialog();

};


#endif //  __PTCURMOVER__
