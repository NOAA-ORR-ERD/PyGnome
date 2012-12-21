/*
 *  GridCurrentMover_c.h
 *  gnome
 *
 *  Created by Generic Programmer on 12/1/11.
 *  Copyright 2011 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef __GridCurrentMover_c__
#define __GridCurrentMover_c__

#include "Basics.h"
#include "TypeDefs.h"
#include "CurrentMover_c.h"
#include "ExportSymbols.h"

#ifndef pyGNOME
#include "GridVel.h"
#include "TimeGridVel.h"
#else
#include "GridVel_c.h"
#define TGridVel GridVel_c
#include "TimeGridVel_c.h"
#define TimeGridVel TimeGridVel_c
#define TMap Map_c
#endif

typedef struct {
	char		pathName[kMaxNameLen];
	char		userName[kPtCurUserNameLen]; // user name for the file, or short file name
	double 	alongCurUncertainty;	
	double 	crossCurUncertainty;	
	double 	uncertMinimumInMPS;	
	double 	curScale;	// use value from file? 	
	double 	startTimeInHrs;	
	double 	durationInHrs;	
	//
	//long		maxNumDepths;
	short		gridType;
	//
	Boolean 	bShowGrid;
	Boolean 	bShowArrows;
	Boolean	bUncertaintyPointOpen;
	double 	arrowScale;
	double 	arrowDepth;	// depth level where velocities will be shown
} GridCurrentVariables;

class GNOMEDLL_API GridCurrentMover_c : virtual public CurrentMover_c {

public:
	GridCurrentVariables fVar;
	Boolean fIsOptimizedForStep;
	TimeGridVel *timeGrid;
	
	Boolean fAllowVerticalExtrapolationOfCurrents;
	float	fMaxDepthForExtrapolation;
	
#ifndef pyGNOME
	GridCurrentMover_c (TMap *owner, char *name);
#endif
	//GridCurrentMover_c () {timeGrid = 0; fLESetSizesH = 0; fUncertaintyListH = 0;} 
	GridCurrentMover_c (); 
	
	virtual ClassID 	GetClassID () { return TYPE_GRIDCURRENTMOVER; }
	virtual Boolean	IAm(ClassID id) { if(id==TYPE_GRIDCURRENTMOVER) return TRUE; return CurrentMover_c::IAm(id); }

	virtual OSErr		AddUncertainty(long setIndex, long leIndex,VelocityRec *patVelocity,double timeStep,Boolean useEddyUncertainty);
	VelocityRec			GetPatValue (WorldPoint p);
	VelocityRec 		GetScaledPatValue(const Seconds& model_time, WorldPoint p,Boolean * useEddyUncertainty);//JLM 5/12/99
	virtual float		GetArrowDepth() {return fVar.arrowDepth;}
	
	virtual WorldRect	GetGridBounds(){return timeGrid->GetGridBounds();}	
	//virtual OSErr		InitMover (TimeGridVel *grid); 
	//virtual void		SetTimeGrid(TimeGridVel *newTimeGrid) {timeGrid = newTimeGrid;}
	void		SetTimeGrid(TimeGridVel *newTimeGrid) {timeGrid = newTimeGrid;}

	virtual OSErr 		PrepareForModelRun(); 
	virtual WorldPoint3D       GetMove(const Seconds& model_time, Seconds timeStep,long setIndex,long leIndex,LERec *thisLE,LETYPE leType);
	virtual OSErr 		PrepareForModelStep(const Seconds&, const Seconds&, bool, int numLESets, int* LESetsSizesList); 
	virtual void 		ModelStepIsDone();
	
			OSErr		TextRead(char *path,char *topFilePath);

			OSErr		get_move(int n, unsigned long model_time, unsigned long step_len, WorldPoint3D* ref, WorldPoint3D* delta, short* LE_status, LEType spillType, long spill_ID);

};

#undef TMap
#endif
