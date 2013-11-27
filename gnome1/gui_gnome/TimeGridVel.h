
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
	virtual				~TimeGridVel () { Dispose (); }
	//virtual void		Dispose ();

	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDVEL; }
	virtual Boolean	IAm(ClassID id) { return(id==TYPE_TIMEGRIDVEL);}
	
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	//OSErr 				ReadInputFileNames(char *fileNamesPath);
	virtual Boolean		VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth) {return 0;}
	
	virtual void Draw(Rect r, WorldRect view,double refScale,
					  double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor)=0;
	//virtual	OSErr 	ExportTopology(char* path) {return 0;}
};

class TimeGridVelRect : virtual public TimeGridVelRect_c, public TimeGridVel
{
public:

	TimeGridVelRect();
	virtual ~TimeGridVelRect () { Dispose (); }
	//virtual void		Dispose ();
	
	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDVELRECT; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEGRIDVELRECT) return TRUE; return TimeGridVel::IAm(id); }
	
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
	virtual				~TimeGridVelCurv () { Dispose (); }
	//virtual void		Dispose ();

	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDVELCURV; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEGRIDVELCURV) return TRUE; return TimeGridVelRect::IAm(id); }

	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	Boolean			VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth);
	
	//virtual	OSErr 	ReadTopology(char* path);
	//virtual	OSErr 	ExportTopology(char* path);

	virtual void Draw(Rect r, WorldRect view,double refScale,
					  double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor);
};

class TimeGridVelTri : virtual public TimeGridVelTri_c, public TimeGridVelCurv
{
public:
						TimeGridVelTri();
	virtual				~TimeGridVelTri () { Dispose (); }
	//virtual void		Dispose ();

	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDVELTRI; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEGRIDVELTRI) return TRUE; return TimeGridVelCurv::IAm(id); }

	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	Boolean			VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth);
	
	//virtual	OSErr 	ReadTopology(char* path);
	//virtual	OSErr 	ExportTopology(char* path);

	virtual void Draw(Rect r, WorldRect view,double refScale,
					  double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor);
	
};

class TimeGridCurRect : virtual public TimeGridCurRect_c, public TimeGridVel
{
public:
	
	TimeGridCurRect();
	virtual ~TimeGridCurRect () { Dispose (); }
	//virtual void		Dispose ();
	
	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDCURRECT; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEGRIDCURRECT) return TRUE; return TimeGridVel::IAm(id); }
	
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
	virtual ~TimeGridCurTri () { Dispose (); }
	//virtual void		Dispose ();
	
	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDCURTRI; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEGRIDCURTRI) return TRUE; return TimeGridCurRect::IAm(id); }

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
	virtual ~TimeGridWindRect () { Dispose (); }
	//virtual void		Dispose ();
	
	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDWINDRECT; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEGRIDWINDRECT) return TRUE; return TimeGridVel::IAm(id); }
	
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
	virtual ~TimeGridWindCurv () { Dispose (); }
	//virtual void		Dispose ();
	
	virtual ClassID 	GetClassID () { return TYPE_TIMEGRIDWINDCURV; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_TIMEGRIDWINDCURV) return TRUE;  return TimeGridWindRect::IAm(id); }
	
	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	Boolean			VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr, double arrowDepth);
	
	//virtual	OSErr 	ReadTopology(char* path);
	//virtual	OSErr 	ExportTopology(char* path);
	
	virtual void Draw(Rect r, WorldRect view,double refScale,
					  double arrowScale,double arrowDepth, Boolean bDrawArrows, Boolean bDrawGrid, RGBColor arrowColor);
};

/*class TimeGridWindRectASCII : virtual public TimeGridCWindRectASCII_c, public TimeGridVel
{
public:
	
	TimeGridWindRectASCII();
	virtual ~TimeGridWindRectASCII () { Dispose (); }
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
