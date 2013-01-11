
#ifndef __TIMEGRIDVEL__
#define __TIMEGRIDVEL__

#include "TimeGridVel_c.h"

#include "GridVel.h"
//#include "PtCurMap.h"
#include "TWindMover.h"

class TimeGridVel : virtual public TimeGridVel_c
{
public:
						TimeGridVel();
						~TimeGridVel () { Dispose (); }
	virtual void		Dispose ();

	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	//OSErr 				ReadInputFileNames(char *fileNamesPath);
	virtual Boolean		VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth) {return 0;}
	
	virtual void Draw(Rect r, WorldRect view,double refScale,
					  double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor)=0;
	virtual	OSErr 	ExportTopology(char* path) {return 0;}
};

class TimeGridVelRect : virtual public TimeGridVelRect_c, public TimeGridVel
{
public:

	TimeGridVelRect();
	~TimeGridVelRect () { Dispose (); }
	virtual void		Dispose ();
	
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	Boolean			VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth);
	
	virtual void Draw(Rect r, WorldRect view,double refScale,
					  double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor); 
};

class TimeGridVelCurv : virtual public TimeGridVelCurv_c, public TimeGridVelRect
{
public:
						TimeGridVelCurv();
						~TimeGridVelCurv () { Dispose (); }
	virtual void		Dispose ();

	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	Boolean			VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth);
	
	//virtual	OSErr 	ReadTopology(char* path);
	virtual	OSErr 	ExportTopology(char* path);

	virtual void Draw(Rect r, WorldRect view,double refScale,
					  double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor);
};

class TimeGridVelTri : virtual public TimeGridVelTri_c, public TimeGridVelCurv
{
public:
						TimeGridVelTri();
						~TimeGridVelTri () { Dispose (); }
	virtual void		Dispose ();

	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	Boolean			VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth);
	
	//virtual	OSErr 	ReadTopology(char* path);
	virtual	OSErr 	ExportTopology(char* path);

	virtual void Draw(Rect r, WorldRect view,double refScale,
					  double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor);
	
};

class TimeGridCurRect : virtual public TimeGridCurRect_c, public TimeGridVel
{
public:
	
	TimeGridCurRect();
	~TimeGridCurRect () { Dispose (); }
	virtual void		Dispose ();
	
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	//OSErr			ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH, char *pathOfInputfile);
	Boolean			VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth);
	
	virtual void Draw(Rect r, WorldRect view,double refScale,
					  double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor); 
};

class TimeGridCurTri : virtual public TimeGridCurTri_c, public TimeGridCurRect
{
public:
	
	TimeGridCurTri();
	~TimeGridCurTri () { Dispose (); }
	virtual void		Dispose ();
	
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	Boolean			VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth);
	
	virtual void Draw(Rect r, WorldRect view,double refScale,
					  double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor); 
};

class TimeGridWindRect : virtual public TimeGridWindRect_c, public TimeGridVel
{
public:

	TimeGridWindRect();
	~TimeGridWindRect () { Dispose (); }
	virtual void		Dispose ();
	
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	Boolean			VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth);
	
	virtual void Draw(Rect r, WorldRect view,double refScale,
					  double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor); 
};

class TimeGridWindCurv : virtual public TimeGridWindCurv_c, public TimeGridWindRect
{
public:
	TimeGridWindCurv();
	~TimeGridWindCurv () { Dispose (); }
	virtual void		Dispose ();
	
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	Boolean			VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth);
	
	//virtual	OSErr 	ReadTopology(char* path);
	virtual	OSErr 	ExportTopology(char* path);
	
	virtual void Draw(Rect r, WorldRect view,double refScale,
					  double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor);
};

/*class TimeGridWindRectASCII : virtual public TimeGridCWindRectASCII_c, public TimeGridVel
{
public:
	
	TimeGridWindRectASCII();
	~TimeGridWindRectASCII () { Dispose (); }
	virtual void		Dispose ();
	
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	//OSErr			ReadInputFileNames(CHARH fileBufH, long *line, long numFiles, PtCurFileInfoH *inputFilesH, char *pathOfInputfile);
	Boolean			VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth);
	
	virtual void Draw(Rect r, WorldRect view,double refScale,
					  double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor); 
};
*/

#endif //  __TIMEGRIDVEL__
