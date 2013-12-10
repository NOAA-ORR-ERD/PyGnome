
#ifndef __PTCURMOVER__
#define __PTCURMOVER__

#include "TCurrentMover.h"
#include "DagTree.h"
#include "my_build_list.h"
#include "GridVel.h"
#include "PtCurMover_c.h"

Boolean IsPtCurFile (char *path);
bool IsPtCurFile(std::vector<std::string> &linesInFile);
//Boolean IsPtCurVerticesHeaderLine(char *s, long* numPts, long* numLandPts);
//OSErr ScanDepth (char *startChar, double *DepthPtr);
//void CheckYear(short *year);

#define PTCUR_DELIM_STR " \t"

class PtCurMover : virtual public PtCurMover_c,  public TCurrentMover
{
	public:
							PtCurMover (TMap *owner, char *name);
						   ~PtCurMover () { Dispose (); }

		virtual OSErr		InitMover ();

		virtual void		Dispose ();
	
		virtual ClassID 	GetClassID () { return TYPE_PTCURMOVER; }
		virtual Boolean	IAm(ClassID id) { if(id==TYPE_PTCURMOVER) return TRUE; return TCurrentMover::IAm(id); }

		virtual WorldPoint3D	GetMove (const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
		VelocityRec			GetMove3D(InterpolationVal interpolationVal,float depth);
		virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);

		// I/O methods
		virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
		virtual OSErr 		Write (BFPB *bfpb); // write to  current position

		virtual OSErr		TextRead(char *path, TMap **newMap);
		OSErr 				ReadPtCurVertices(CHARH fileBufH,long *line,LongPointHdl *pointsH, FLOATH *totalDepth,char* errmsg,long numPoints);
		OSErr 				ReadHeaderLine(char *s);
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


#endif //  __PTCURMOVER__
