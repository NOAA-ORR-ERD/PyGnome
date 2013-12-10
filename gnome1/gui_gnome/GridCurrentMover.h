
#ifndef __GRIDCURRENTMOVER__
#define __GRIDCURRENTMOVER__

#include "GridCurrentMover_c.h"


enum {
	I_GRIDCURRENTNAME = 0 ,
	I_GRIDCURRENTACTIVE, 
	I_GRIDCURRENTGRID, 
	I_GRIDCURRENTARROWS,
	I_GRIDCURRENTSCALE,
	I_GRIDCURRENTUNCERTAINTY,
	I_GRIDCURRENTSTARTTIME,
	I_GRIDCURRENTDURATION, 
	I_GRIDCURRENTALONGCUR,
	I_GRIDCURRENTCROSSCUR,
};


class GridCurrentMover : virtual public GridCurrentMover_c,  public TCurrentMover
{

public:
	//Boolean bShowDepthContours;
	//Boolean bShowDepthContourLabels;
	//Rect fLegendRect;
	Boolean fArrowScale;
	Boolean fArrowDepth;
	Boolean bShowGrid;
	Boolean bShowArrows;
	Boolean	bUncertaintyPointOpen;


						GridCurrentMover (TMap *owner, char *name);
	virtual				~GridCurrentMover () { Dispose (); }
	//virtual void		Dispose ();

	virtual ClassID 	GetClassID () { return TYPE_GRIDCURRENTMOVER; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_GRIDCURRENTMOVER) return TRUE; return TCurrentMover::IAm(id); }
	
	virtual OSErr		InitMover (TimeGridVel *grid); 

	// I/O methods
	virtual OSErr 		Read (BFPB *bfpb); 	// read from current position
	virtual OSErr 		Write (BFPB *bfpb); // write to  current position
	
	// list display methods
	virtual OSErr 		CheckAndPassOnMessage(TModelMessage *message);
	
	//virtual void		DrawContourScale(Rect r, WorldRect view);
	virtual void		Draw (Rect r, WorldRect view);
	virtual Boolean	DrawingDependsOnTime(void);
	virtual Boolean 	VelocityStrAtPoint(WorldPoint3D wp, char *diagnosticStr);
	virtual float		GetArrowDepth() {return fArrowDepth;}
	
	virtual long		GetListLength ();
	virtual ListItem 	GetNthListItem (long n, short indent, short *style, char *text);
	virtual Boolean 	ListClick (ListItem item, Boolean inBullet, Boolean doubleClick);
	virtual Boolean 	FunctionEnabled (ListItem item, short buttonID);
	//virtual OSErr 		AddItem (ListItem item);
	virtual OSErr 		SettingsItem (ListItem item);
	virtual OSErr 		DeleteItem (ListItem item);
	
	virtual	OSErr 	ExportTopology(char* path) {return timeGrid->ExportTopology(path);}

	virtual OSErr 		SettingsDialog();
	

};


#endif //  __GRIDCURRENTMOVER__
